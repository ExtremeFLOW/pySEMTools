# script-version: 2.0
# Catalyst state generated using paraview version 6.0.1
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=[2200, 1171],
    CenterOfRotation=[0.0, 0.0, 0.5],
    CameraPosition=[0.3087330117565116, -1.7780451991279786, 1.2415733431390492],
    CameraFocalPoint=[-1.9020516812546384e-18, -1.8545003892232722e-17, 0.49999999999999994],
    CameraViewUp=[0.17093908901864524, 0.40440530996644536, 0.898463228583351],
    CameraFocalDisk=1.0,
    CameraParallelScale=0.5049752470656473,
)

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'VTKHDF Reader'
rbc00001vtkhdf = VTKHDFReader(registrationName='rbc00001.vtkhdf', FileName=['/home/adperez/software/pySEMTools/examples/data/rbc00001.vtkhdf'])
rbc00001vtkhdf.PointArrayStatus = ['p', 't', 'u', 'v', 'w']

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=rbc00001vtkhdf)
contour1.Set(
    ContourBy=['POINTS', 'w'],
    Isosurfaces=[-0.132496640086174, -0.044175898035367325, 0.04414484401543936, 0.13246558606624603],
)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from contour1
contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'w'
wLUT = GetColorTransferFunction('w')
wLUT.Set(
    RGBPoints=GenerateRGBPoints(
        range_min=-0.132496640086174,
        range_max=0.13246558606624603,
    ),
    ScalarRangeInitialized=1.0,
)

# trace defaults for the display properties.
contour1Display.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', 'w'],
    LookupTable=wLUT,
    SelectNormalArray='Normals',
)

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [-0.0441758967936039, 0.0, 0.5, 0.0, 0.13246558606624603, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [-0.0441758967936039, 0.0, 0.5, 0.0, 0.13246558606624603, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for wLUT in view renderView1
wLUTColorBar = GetScalarBar(wLUT, renderView1)
wLUTColorBar.Set(
    Title='w',
    ComponentTitle='',
)

# set color bar visibility
wLUTColorBar.Visibility = 1

# show color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'w'
wPWF = GetOpacityTransferFunction('w')
wPWF.Set(
    Points=[-0.132496640086174, 0.0, 0.5, 0.0, 0.13246558606624603, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
# init the 'PNG' selected for 'Writer'
pNG1.Writer.Set(
    FileName='RenderView1_{timestep:06d}{camera}.png',
    ImageResolution=[2200, 1171],
    Format='PNG',
)

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.ExtractsOutputDirectory = '/home/adperez/software/pySEMTools/scripts/4-insitu_visualization'

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from paraview.simple import SaveExtractsUsingCatalystOptions
    # Code for non in-situ environments; if executing in post-processing
    # i.e. non-Catalyst mode, let's generate extracts using Catalyst options
    SaveExtractsUsingCatalystOptions(options)
