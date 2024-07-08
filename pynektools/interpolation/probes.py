import json
import numpy as np
import csv
import json
from ..ppymech.neksuite import preadnek, pwritenek
from .interpolator import interpolator_c
from .mpi_ops import gather_in_root, scatter_from_root

NoneType = type(None)


class probes_c:

    class io_data:
        def __init__(self, params_file):
            self.casename = params_file["casename"]
            self.dataPath = params_file["dataPath"]
            self.index = params_file["first_index"]

    def __init__(
        self,
        comm,
        filename=None,
        probes=None,
        msh=None,
        write_coords=True,
        progress_bar=False,
        modal_search=True,
        communicate_candidate_pairs=True,
        elem_percent_expansion=0.01,
    ):

        rank = comm.Get_rank()
        size = comm.Get_size()

        # Open input file
        if type(filename) != NoneType:
            f = open(filename, "r")
            params_file = json.loads(f.read())
            params_file_str = json.dumps(params_file, indent=4)
            f.close()
            self.output_data = self.io_data(params_file["case"]["IO"]["output_data"])
            self.params_file = params_file
        else:
            default_output = {}
            default_output["dataPath"] = "./"
            default_output["casename"] = "interpolated_fields.csv"
            default_output["first_index"] = 0
            self.output_data = self.io_data(default_output)

        if type(probes) == NoneType:
            self.probes_data = self.io_data(params_file["case"]["IO"]["probe_data"])
            # Read probes
            probe_fname = self.probes_data.dataPath + self.probes_data.casename
            if rank == 0:
                file = open(probe_fname)
                self.probes = np.array(list(csv.reader(file)), dtype=np.double)
            else:
                self.probes = None
        else:
            # Assign probes to the required shape
            if rank == 0:
                self.probes = probes
            else:
                self.probes = None

        if type(msh) == NoneType:
            # read mesh data

            self.mesh_data = self.io_data(params_file["case"]["IO"]["mesh_data"])

            msh_fld_fname = (
                self.mesh_data.dataPath
                + self.mesh_data.casename
                + "0.f"
                + str(self.mesh_data.index).zfill(5)
            )
            mesh_data = preadnek(msh_fld_fname, comm)
            self.x, self.y, self.z = get_coordinates_from_hexadata(mesh_data)
            del mesh_data

        else:
            self.x = msh.x
            self.y = msh.y
            self.z = msh.z

        # Initialize the interpolator
        self.itp = interpolator_c(
            self.x,
            self.y,
            self.z,
            self.probes,
            comm,
            progress_bar,
            modal_search=modal_search,
        )

        # Scatter the probes to all ranks
        self.itp.scatter_probes_from_io_rank(0, comm)

        # Find where the point in each rank should be
        if comm.Get_rank() == 0:
            print("finding points")
        if communicate_candidate_pairs == False:
            if comm.Get_rank() == 0:
                print("Not identifying rank pairs - only works with powers of 2 ranks")
            self.itp.find_points(comm, elem_percent_expansion=elem_percent_expansion)
        else:
            if comm.Get_rank() == 0:
                print("Identifying rank pairs - Might be slower")
            self.itp.find_points_comm_pairs(
                comm,
                communicate_candidate_pairs=communicate_candidate_pairs,
                elem_percent_expansion=elem_percent_expansion,
            )

        # Gather probes to rank 0 again
        self.itp.gather_probes_to_io_rank(0, comm)
        if comm.Get_rank() == 0:
            print("found data")

        # Redistribute the points
        self.itp.redistribute_probes_to_owners_from_io_rank(0, comm)
        if comm.Get_rank() == 0:
            print("redistributed data")

        self.output_fname = self.output_data.dataPath + self.output_data.casename
        if write_coords:

            if rank == 0:
                # Write the coordinates
                write_csv(self.output_fname, self.probes, "w")

                # Write out a file with the points with warnings
                indices = np.where(self.itp.err_code != 1)[0]
                # Write the points with warnings in a json file
                point_warning = {}
                # point_warning["mesh_file_path"] = msh_fld_fname
                point_warning["output_file_path"] = self.output_fname
                i = 0
                for point in indices:
                    point_warning[i] = {}
                    point_warning[i]["id"] = int(point)
                    point_warning[i]["xyz"] = [
                        float(self.itp.probes[point, 0]),
                        float(self.itp.probes[point, 1]),
                        float(self.itp.probes[point, 2]),
                    ]
                    point_warning[i]["rst"] = [
                        float(self.itp.probes_rst[point, 0]),
                        float(self.itp.probes_rst[point, 1]),
                        float(self.itp.probes_rst[point, 2]),
                    ]
                    # point_warning[i]["local_el_owner"] = int(self.itp.el_owner[point])
                    point_warning[i]["global_el_owner"] = int(
                        self.itp.glb_el_owner[point]
                    )
                    point_warning[i]["error_code"] = int(self.itp.err_code[point])
                    point_warning[i]["test_pattern"] = float(
                        self.itp.test_pattern[point]
                    )

                    i += 1

                params_file_str = json.dumps(point_warning, indent=4)
                json_output_fname = (
                    self.output_data.dataPath
                    + "warning_points_"
                    + self.output_data.casename[:-4]
                    + ".json"
                )
                with open(json_output_fname, "w") as outfile:
                    outfile.write(params_file_str)

        return

    def read_fld_file(self, file_number, comm):

        self.fld_data = self.io_data(self.params_file["case"]["IO"]["fld_data"])
        self.list_of_fields = self.params_file["case"]["interpolate_fields"][
            "field_type"
        ]
        self.list_of_qoi = self.params_file["case"]["interpolate_fields"]["field"]
        self.number_of_files = self.params_file["case"]["interpolate_fields"][
            "number_of_files"
        ]
        self.number_of_fields = len(self.list_of_fields)

        # read field data
        fld_fname = (
            self.fld_data.dataPath
            + self.fld_data.casename
            + "0.f"
            + str(self.fld_data.index + file_number).zfill(5)
        )

        if comm.Get_rank() == 0:
            print("Reading file: {}".format(fld_fname))
        fld_data = preadnek(fld_fname, comm)

        return fld_data

    def interpolate_from_hexadata_and_writecsv(self, fld_data, comm, mode="rst"):

        self.fld_data = self.io_data(self.params_file["case"]["IO"]["fld_data"])
        self.list_of_fields = self.params_file["case"]["interpolate_fields"][
            "field_type"
        ]
        self.list_of_qoi = self.params_file["case"]["interpolate_fields"]["field"]
        self.number_of_files = self.params_file["case"]["interpolate_fields"][
            "number_of_files"
        ]
        self.number_of_fields = len(self.list_of_fields)

        # Allocate interpolated fields
        self.my_interpolated_fields = np.zeros(
            (self.itp.my_probes.shape[0], self.number_of_fields + 1), dtype=np.double
        )
        if comm.Get_rank() == 0:
            self.interpolated_fields = np.zeros(
                (self.probes.shape[0], self.number_of_fields + 1), dtype=np.double
            )
        else:
            self.interpolated_fields = None

        # Set the time
        self.my_interpolated_fields[:, 0] = fld_data.time

        for i in range(0, self.number_of_fields):
            field = get_field_from_hexadata(
                fld_data, self.list_of_fields[i], self.list_of_qoi[i]
            )

            if mode == "rst":
                self.my_interpolated_fields[:, i + 1] = (
                    self.itp.interpolate_field_from_rst(field)[:]
                )

            print(
                "Rank: {}, interpolated field: {}:{}".format(
                    comm.Get_rank(), self.list_of_fields[i], self.list_of_qoi[i]
                )
            )

        # Write to the csv file
        root = 0
        sendbuf = self.my_interpolated_fields.reshape(
            (self.my_interpolated_fields.size)
        )
        recvbuf, _ = gather_in_root(sendbuf, root, np.double, comm)

        if type(recvbuf) != NoneType:
            tmp = recvbuf.reshape(
                (
                    int(recvbuf.size / (self.number_of_fields + 1)),
                    self.number_of_fields + 1,
                )
            )
            # IMPORTANT - After gathering, remember to sort to the way that it should be in rank zero to write values out (See the scattering routine if you forget this)
            # The reason for this is that to scatter we need the data to be contigous. You sort to make sure that the data from each rank is grouped.
            self.interpolated_fields[self.itp.sort_by_rank] = tmp

            # Write data in the csv file
            write_csv(self.output_fname, self.interpolated_fields, "a")

        return

    def interpolate_from_field_list(self, t, field_list, comm, write_data=True):

        self.number_of_fields = len(field_list)

        # Allocate interpolated fields
        self.my_interpolated_fields = np.zeros(
            (self.itp.my_probes.shape[0], self.number_of_fields + 1), dtype=np.double
        )
        if comm.Get_rank() == 0:
            self.interpolated_fields = np.zeros(
                (self.probes.shape[0], self.number_of_fields + 1), dtype=np.double
            )
        else:
            self.interpolated_fields = None

        # Set the time
        self.my_interpolated_fields[:, 0] = t

        i = 0
        for field in field_list:

            self.my_interpolated_fields[:, i + 1] = self.itp.interpolate_field_from_rst(
                field
            )[:]

            if comm.Get_rank() == 0:
                print("Rank: {}, interpolated field: {}".format(comm.Get_rank(), i))

            i += 1

        # Gather in rank zero for processing
        root = 0
        sendbuf = self.my_interpolated_fields.reshape(
            (self.my_interpolated_fields.size)
        )
        recvbuf, _ = gather_in_root(sendbuf, root, np.double, comm)

        if type(recvbuf) != NoneType:
            tmp = recvbuf.reshape(
                (
                    int(recvbuf.size / (self.number_of_fields + 1)),
                    self.number_of_fields + 1,
                )
            )
            # IMPORTANT - After gathering, remember to sort to the way that it should be in rank zero to write values out (See the scattering routine if you forget this)
            # The reason for this is that to scatter we need the data to be contigous. You sort to make sure that the data from each rank is grouped.
            self.interpolated_fields[self.itp.sort_by_rank] = tmp

            # Write data in the csv file
            if write_data:
                write_csv(self.output_fname, self.interpolated_fields, "a")

        return


def get_coordinates_from_hexadata(data):

    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    x = np.zeros((nelv, lz, ly, lx), dtype=np.double)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    for e in range(0, nelv):
        x[e, :, :, :] = data.elem[e].pos[0, :, :, :]
        y[e, :, :, :] = data.elem[e].pos[1, :, :, :]
        z[e, :, :, :] = data.elem[e].pos[2, :, :, :]

    return x, y, z


def write_csv(fname, data, mode):
    """write point positions to the file"""

    string = "writing .csv file as " + fname
    print(string)

    # open file
    outfile = open(fname, mode)

    writer = csv.writer(outfile)

    for il in range(data.shape[0]):
        data_pos = data[il, :]
        writer.writerow(data_pos)

    return


def get_field_from_hexadata(data, prefix, qoi):

    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    field = np.zeros((nelv, lz, ly, lx), dtype=np.double)

    if prefix == "vel":
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].vel[qoi, :, :, :]

    if prefix == "pres":
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].pres[0, :, :, :]

    if prefix == "temp":
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].temp[0, :, :, :]

    if prefix == "scal":
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].scal[qoi, :, :, :]

    return field
