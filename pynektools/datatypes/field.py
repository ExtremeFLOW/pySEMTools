import numpy as np

NoneType = type(None)


class field_c:
    def __init__(self, comm, data=None):

        self.fields = {}
        self.fields["vel"] = []
        self.fields["pres"] = []
        self.fields["temp"] = []
        self.fields["scal"] = []
        if type(data) != NoneType:
            vars = data.var
            self.vel_fields = vars[1]
            self.pres_fields = vars[2]
            self.temp_fields = vars[3]
            self.scal_fields = vars[4]

            # Read the full fields
            for qoi in range(0, self.vel_fields):
                prefix = "vel"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            for qoi in range(0, self.pres_fields):
                prefix = "pres"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            for qoi in range(0, self.temp_fields):
                prefix = "temp"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            for qoi in range(0, self.scal_fields):
                prefix = "scal"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            self.t = data.time

    def update_vars(self):
        self.vel_fields = len(self.fields["vel"])
        self.pres_fields = len(self.fields["pres"])
        self.temp_fields = len(self.fields["temp"])
        self.scal_fields = len(self.fields["scal"])


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
