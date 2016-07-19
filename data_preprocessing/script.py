USERPATH = os.path.expanduser("~")
print(USERPATH)
import time
import SimpleITK as sitk

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-k", "--k", help="case index", type=int)
parser.add_argument("-s", "--s", help="spacing", type=str)
args = parser.parse_args()


paths = []
needPath = []
cases = []
spacing = [int(x) for x in args.s.split(',')]
def createDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return 0

createDir(USERPATH + '/Projects/LabelMaps_%d-%d-%d/'%(spacing[0], spacing[1], spacing[2]))

with open(USERPATH + '/Dropbox/GYN Cases/scenes.txt') as f:
    lines = f.readlines()

#
with open(USERPATH + '/Dropbox/GYN Cases/760CYNeedles.txt') as g:
    needles = g.readlines()

#
for line in lines:
    paths.append(USERPATH + '/Dropbox/GYN Cases' + line[1:-1])
    cases.append(int(line.lstrip('./Case  ')[1:3]))

#
for needle in needles:
    needPath.append(USERPATH + '/Dropbox/GYN Cases' + needle[1:-1])

#  
def getNeedles(case):
    return [s for s in needPath if "%03d"%int(case) in s]

#
def labelMapFromNeedle(inputVolume, needleID, value, caseNumber, name):
    '''
    Convert a needle to a labelmap, save it and remove the node from the scene.
    '''
    outputLabelMap = slicer.vtkMRMLLabelMapVolumeNode()
    slicer.mrmlScene.AddNode(outputLabelMap)
    params = {'sampleDistance': 1, 'labelValue': value, 'InputVolume': inputVolume.GetID(),
              'surface': needleID, 'OutputVolume': outputLabelMap.GetID()}
    slicer.cli.run(slicer.modules.modeltolabelmap, None, params, wait_for_completion=True)
    # slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed").SetUseLabelOutline(True)
    # slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow").SetUseLabelOutline(True)
    # slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeRed").RotateToVolumePlane(outputLabelMap)
    # slicer.mrmlScene.GetNodeByID("vtkMRMLSliceNodeYellow").RotateToVolumePlane(outputLabelMap)
    slicer.util.saveNode(outputLabelMap, USERPATH + '/Projects/LabelMaps_%d-%d-%d/%d/needle-%s.nrrd'%(spacing[0], spacing[1], spacing[2], caseNumber,name))
    # slicer.mrmlScene.RemoveAllObservers()
    slicer.mrmlScene.RemoveNodeReferences(outputLabelMap)
    slicer.mrmlScene.RemoveNode(outputLabelMap)
    return 0

#
def get_resized_img(k, data_type = sitk.sitkFloat32):
    """
    This function resizes an image to a fixed shape.
    If data type is sitkFloat32 a linear interpolation is used, otherwise nearest neighbor interpolation is used.
    """
    img = sitk.ReadImage(USERPATH + '/Projects/LabelMaps/%d/case.nrrd'%(k))
    size = img.GetSize()
    ratio = [1.0/i for i in img.GetSpacing()]
    new_size = [int(size[i]/ratio[i]) for i in range(3)]
    
    rimage = sitk.Image(new_size, data_type)
    rimage.SetSpacing((spacing[0], spacing[1], spacing[2]))
    rimage.SetOrigin(img.GetOrigin())
    tx = sitk.Transform()
    
    interp = sitk.sitkLinear
    if data_type == sitk.sitkInt16:
        interp = sitk.sitkNearestNeighbor
    
    new_image = sitk.Resample(img, rimage, tx, interp, data_type)
    dirPath = USERPATH + '/Projects/LabelMaps_%d-%d-%d/%d/'%(spacing[0], spacing[1], spacing[2], k)
    createDir(dirPath)
    filename = USERPATH + '/Projects/LabelMaps_%d-%d-%d/%d/case.nrrd'%(spacing[0], spacing[1], spacing[2], k)
    sitk.WriteImage( new_image, filename )
    return filename

#
def extract(k):
    '''
    Extract needles of just one case.
    '''
    slicer.util.loadScene(paths[k])
    #nodes = slicer.util.getNodes("manual-seg*")
    ndls = getNeedles(cases[k])
    imgPath = get_resized_img(cases[k])
    slicer.util.loadVolume(imgPath)
    backgroundNode = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetBackgroundLayer().GetVolumeNode()
    slicer.util.saveNode(backgroundNode, USERPATH + '/Projects/LabelMaps_%d-%d-%d/%d/case.nrrd'%(spacing[0], spacing[1], spacing[2], cases[k]))
    for i, ndl in enumerate(ndls):
        try:
            name = ndl.split('_')[-1].split('.')[0]
            _, node = slicer.util.loadModel(ndl, 'ndl')
            if node:
                labelMapFromNeedle(backgroundNode, node.GetID(), i+1, cases[k], name)
                slicer.mrmlScene.RemoveNode(node)
        except:
            pass
    quit()
    return 0 

extract(args.k)
#print(spacing)
