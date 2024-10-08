###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Feb 12th 2021                                                 #
###########################################################################################

####################################################################################################
#                                                                                                  #
# THE CURRENT VERSION WAS UPDATED WITH A VISUAL INTERFACE, INCLUDING MORE METRICS AND SUPPORTING   #
# OTHER FILE FORMATS. PLEASE ACCESS IT ACCESSED AT:                                                #
#                                                                                                  #
# https://github.com/rafaelpadilla/review_object_detection_metrics                                 #
#                                                                                                  #
# @Article{electronics10030279,                                                                    #
#     author         = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto,    #
#                       Sergio L. and da Silva, Eduardo A. B.},                                    #
#     title          = {A Comparative Analysis of Object Detection Metrics with a Companion        #
#                       Open-Source Toolkit},                                                      #
#     journal        = {Electronics},                                                              #
#     volume         = {10},                                                                       #
#     year           = {2021},                                                                     #
#     number         = {3},                                                                        #
#     article-number = {279},                                                                      #
#     url            = {https://www.mdpi.com/2079-9292/10/3/279},                                  #
#     issn           = {2079-9292},                                                                #
#     doi            = {10.3390/electronics10030279}, }                                            #
####################################################################################################

####################################################################################################
# If you use this project, please consider citing:                                                 #
#                                                                                                  #
# @INPROCEEDINGS {padillaCITE2020,                                                                 #
#    author    = {R. {Padilla} and S. L. {Netto} and E. A. B. {da Silva}},                         #
#    title     = {A Survey on Performance Metrics for Object-Detection Algorithms},                #
#    booktitle = {2020 International Conference on Systems, Signals and Image Processing (IWSSIP)},#
#    year      = {2020},                                                                           #
#    pages     = {237-242},}                                                                       #
#                                                                                                  #
# This work is published at: https://github.com/rafaelpadilla/Object-Detection-Metrics             #
####################################################################################################

import argparse
import glob
import os
import shutil
import sys
import cv2
import pandas as pd
from pandas.plotting import table
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import json
from tqdm import tqdm

import _init_paths
from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.Evaluator import *
from lib.utils import BBFormat


# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append('argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' %
                      argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append('%s. It must be in the format \'width,height\' (e.g. \'600,400\')' %
                          errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, currentPath, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(directory,
                     isGT,
                     bbFormat,
                     coordType,
                     allBoundingBoxes=None,
                     allClasses=None,
                     img_dir=None,
                     imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in tqdm(files):
        nameOfImage = f.replace(".txt", "")
        if (img_dir != "") and (img_dir != None):
            img_path = f"{img_dir}/{nameOfImage}.jpg"
            image = cv2.imread(str(img_path))
            img_h, img_w = image.shape[:2]
            img_area = img_h * img_w
        else:
            img_path = ""
            img_h, img_w = 0, 0
            img_area = 0
        
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(nameOfImage,
                                 idClass,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coordType,
                                 imgSize,
                                 BBType.GroundTruth,
                                 format=bbFormat,
                                 imageArea=img_area)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(nameOfImage,
                                 idClass,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coordType,
                                 imgSize,
                                 BBType.Detected,
                                 confidence,
                                 format=bbFormat,
                                 imageArea=img_area)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


def main():
    # Get current path to set default folders
    currentPath = os.path.dirname(os.path.abspath(__file__))
    VERSION = '0.2 (beta)'
    with open('message.txt', 'r') as f:
        message = f'\n\n{f.read()}\n\n'
    # print(message)

    parser = argparse.ArgumentParser(
        prog='Object Detection Metrics - Pascal VOC',
        description=
        f'{message}\nThis project applies the most popular metrics used to evaluate object detection '
        'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
        'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
        epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
    # Positional arguments
    # Mandatory
    parser.add_argument('-gt',
                        '--gtfolder',
                        dest='gtFolder',
                        default=os.path.join(currentPath, 'groundtruths'),
                        metavar='',
                        help='folder containing your ground truth bounding boxes')
    parser.add_argument('-det',
                        '--detfolder',
                        dest='detFolder',
                        default=os.path.join(currentPath, 'detections'),
                        metavar='',
                        help='folder containing your detected bounding boxes')
    # Optional
    parser.add_argument('-t',
                        '--threshold',
                        dest='iouThreshold',
                        type=float,
                        default=0.5,
                        metavar='',
                        help='IOU threshold. Default 0.5')
    parser.add_argument('-gtformat',
                        dest='gtFormat',
                        metavar='',
                        default='xywh',
                        help='format of the coordinates of the ground truth bounding boxes: '
                        '(\'xywh\': <left> <top> <width> <height>)'
                        ' or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument('-detformat',
                        dest='detFormat',
                        metavar='',
                        default='xywh',
                        help='format of the coordinates of the detected bounding boxes '
                        '(\'xywh\': <left> <top> <width> <height>) '
                        'or (\'xyrb\': <left> <top> <right> <bottom>)')
    parser.add_argument('-gtcoords',
                        dest='gtCoordinates',
                        default='abs',
                        metavar='',
                        help='reference of the ground truth bounding box coordinates: absolute '
                        'values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument('-detcoords',
                        default='abs',
                        dest='detCoordinates',
                        metavar='',
                        help='reference of the ground truth bounding box coordinates: '
                        'absolute values (\'abs\') or relative to its image size (\'rel\')')
    parser.add_argument('-imgsize',
                        dest='imgSize',
                        metavar='',
                        help='image size. Required if -gtcoords or -detcoords are \'rel\'')
    parser.add_argument('-sp',
                        '--savepath',
                        dest='savePath',
                        metavar='',
                        help='folder where the plots are saved')
    parser.add_argument('-img',
                        '--imgfolder',
                        dest='imgFolder',
                        metavar='',
                        help='folder containing your images')
    parser.add_argument('-np',
                        '--noplot',
                        dest='showPlot',
                        action='store_false',
                        help='no plot is shown during execution')
    args = parser.parse_args()

    iouThreshold = args.iouThreshold

    # Arguments validation
    errors = []

    # Validate formats
    gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
    detFormat = ValidateFormats(args.detFormat, '-detformat', errors)

    # Groundtruth folder
    if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
        gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', currentPath, errors)
    else:
        # errors.pop()
        gtFolder = os.path.join(currentPath, 'groundtruths')
        if os.path.isdir(gtFolder) is False:
            errors.append('folder %s not found' % gtFolder)

    # Coordinates types
    gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
    detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)

    imgSize = (0, 0)

    if gtCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)

    if detCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)

    # Detection folder
    if ValidateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
        detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', currentPath, errors)
    else:
        # errors.pop()
        detFolder = os.path.join(currentPath, 'detections')
        if os.path.isdir(detFolder) is False:
            errors.append('folder %s not found' % detFolder)

    # Validate savePath
    if args.savePath is not None:
        savePath = ValidatePaths(args.savePath, '-sp/--savepath', currentPath, errors)
    else:
        savePath = os.path.join(currentPath, 'results')
    
    # Image folder
    if args.imgFolder:
        imgFolder = args.imgFolder
    else:
        imgFolder = ""

    # If error, show error messages
    if len(errors) != 0:
        print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                    [-detformat] [-save]""")
        print('Object Detection Metrics: error(s): ')
        [print(e) for e in errors]
        sys.exit()

    # Clear folder and save results
    shutil.rmtree(savePath, ignore_errors=True)
    os.makedirs(savePath)
    # Show plot during execution
    showPlot = args.showPlot

    # print('iouThreshold= %f' % iouThreshold)
    # print('savePath = %s' % savePath)
    # print('gtFormat = %s' % gtFormat)
    # print('detFormat = %s' % detFormat)
    # print('gtFolder = %s' % gtFolder)
    # print('detFolder = %s' % detFolder)
    # print('gtCoordType = %s' % gtCoordType)
    # print('detCoordType = %s' % detCoordType)
    # print('showPlot %s' % showPlot)

    # Get groundtruth boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(gtFolder,
                                                    True,
                                                    gtFormat,
                                                    gtCoordType,
                                                    imgSize=imgSize,
                                                    img_dir=imgFolder
                                                    )
    # Get detected boxes
    allBoundingBoxes, allClasses = getBoundingBoxes(detFolder,
                                                    False,
                                                    detFormat,
                                                    detCoordType,
                                                    allBoundingBoxes,
                                                    allClasses,
                                                    imgSize=imgSize,
                                                    img_dir=imgFolder
                                                    )
    allClasses.sort()

    evaluator = Evaluator()
    validClasses, acc_AP, acc_AP_s, acc_AP_m, acc_AP_l = 0, 0, 0, 0, 0
    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        # Object containing all bounding boxes (ground truths and detections)
        allBoundingBoxes,
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=savePath,
        showGraphic=showPlot)

    with open((os.path.join(savePath, 'results.txt')), 'w') as f:
        f.write('Object Detection Metrics\n')
        f.write('Average Precision (AP), Precision and Recall per class:')

        sum_of_classes = 0
        # each detection is a class
        columns = ['Class','GT','TP','FP','FN','Recall','Precision','AP','AP-S','AP-M','AP-L','iou','mAP', 'mAP-S', 'mAP-M', 'mAP-L']
        data = []

        for metricsPerClass in detections:
            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            ap_s = metricsPerClass['AP-S']
            ap_m = metricsPerClass['AP-M']
            ap_l = metricsPerClass['AP-L']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']
            total_TP = metricsPerClass['total TP']
            total_FP = metricsPerClass['total FP']
            sum_of_classes += totalPositives
            prec_recall_json = os.path.join(savePath, f"{cl}.json")

            if totalPositives > 0:
                validClasses = validClasses + 1
                acc_AP = acc_AP + ap
                acc_AP_s = acc_AP_s + ap_s
                acc_AP_m = acc_AP_m + ap_m
                acc_AP_l = acc_AP_l + ap_l

                prec = ['%.2f' % p for p in precision]
                rec = ['%.2f' % r for r in recall]
                rec_percent = total_TP / totalPositives
                prec_percent = (total_TP + total_FP) and total_TP / \
                    (total_TP + total_FP) or 0
                ap_str = "{0:.2f}%".format(ap * 100)
                ap_s_str = "{0:.2f}%".format(ap_s * 100)
                ap_m_str = "{0:.2f}%".format(ap_m * 100)
                ap_l_str = "{0:.2f}%".format(ap_l * 100)
                # ap_str = "{0:.4f}%".format(ap * 100)
                print(f'Class: {cl}')
                print(f'Ground truth: {totalPositives}')
                print(f'TP: {int(total_TP)}, FP: {int(total_FP)}, FN: {int(totalPositives - total_TP)}, Recall: {rec_percent:.2f}, Precision: {prec_percent:.2f}, AP: {ap_str}, AP-S: {ap_s_str}, AP-M: {ap_m_str}, AP-L: {ap_l_str}')

                with open(prec_recall_json, "w") as j:
                    if len(prec) != 0 or len(rec) != 0:
                        json.dump({cl: [{"precision": p, "recall": r} for p, r in zip(prec, rec)]}, j, indent=4)
                    else:
                        json.dump({cl: [{"precision": "0.00", "recall": "0.00"}]}, j, indent=4)
                
                f.write('\n\nClass: %s' % cl)
                f.write(f'\nGround truth: {totalPositives}, TP: {int(total_TP)}, FP: {int(total_FP)}, FN: {int(totalPositives - total_TP)}, Recall: {rec_percent:.2f}, Precision: {prec_percent:.2f}, AP: {ap_str}, AP-S: {ap_s_str}, AP-M: {ap_m_str}, AP-L: {ap_l_str}')
                f.write('\nPrecision: %s' % prec)
                f.write('\nRecall: %s' % rec)

                elements = [cl, totalPositives,
                            int(total_TP),
                            int(total_FP),
                            int(totalPositives - total_TP),
                            rec_percent,
                            prec_percent,
                            ap,
                            ap_s,
                            ap_m,
                            ap_l
                            ]

            data.append(elements)

    # Make results.csv
    mAP = acc_AP / validClasses
    mAP_s = acc_AP_s / validClasses
    mAP_m = acc_AP_m / validClasses
    mAP_l = acc_AP_l / validClasses

    mAP_str = "{0:.2f}%".format(mAP * 100)
    mAP_s_str = "{0:.2f}%".format(mAP_s * 100)
    mAP_m_str = "{0:.2f}%".format(mAP_m * 100)
    mAP_l_str = "{0:.2f}%".format(mAP_l * 100)

    print(f'Total number of labels: {sum_of_classes}')
    print(f'Iou: {iouThreshold}, mAP: {mAP_str}, mAP-S: {mAP_s_str}, mAP-M: {mAP_m_str}, mAP-L: {mAP_l_str}')
    # f.write(f'\n\n\nIou: {iouThreshold}, mAP: {mAP_str}')

    for i in range(len(data)):
        data[i].extend([iouThreshold, mAP, mAP_s, mAP_m, mAP_l])

    df = pd.DataFrame([sublist[:len(columns)] for sublist in data], columns=columns)
    df = df.drop_duplicates(subset="Class")

    cols = ["Recall", "Precision", "AP", "AP-S", "AP-M", "AP-L", "iou", "mAP", "mAP-S", "mAP-M", "mAP-L"]

    for col in cols:
        df[col] = df[col].map('{:,.2f}'.format)

    df.to_csv(f'{savePath}/results.csv', encoding='utf-8', index=False)

    # Make results_rpa.png and results_tfpn.png
    plt.close()

    rpa_df = df[["Class", "Recall", "Precision", "AP", "AP-S", "AP-M", "AP-L", "mAP", "mAP-S", "mAP-M", "mAP-L"]].set_index("Class").astype(float)
    tfpn_df = df[["Class", "GT", "TP", "FP", "FN"]].set_index("Class").astype(int)

    rpa_vals = np.around(rpa_df.values, 2)

    cl_outcomes = {'white':'#FFFFFF',
                'gray': '#AAA9AD',
                'black':'#313639',
                'purple':'#AD688E',
                'orange':'#D18F77',
                'yellow':'#E8E190',
                'ltgreen':'#CCD9C7',
                'dkgreen':'#96ABA0',
                }

    fig_1, axis_1 = plt.subplots(figsize=(8,5))
    axis_1.set(frame_on=False)
    axis_1.axis('off')

    table(axis_1, rpa_df, 
                colWidths = [0.25]*len(rpa_df.columns),
                cellColours=plt.cm.RdYlGn(rpa_vals),
                rowColours=np.full(len(rpa_df.index), cl_outcomes['ltgreen']),
                colColours=np.full(len(rpa_df.columns), cl_outcomes['ltgreen']),
                loc = 'best', edges = 'closed', cellLoc = 'center'
                ).auto_set_column_width(col=list(range(len(rpa_df.columns))))

    if savePath is not None:
        plt.savefig(os.path.join(savePath, 'results_rpa.png'), dpi=100)
        plt.close()

    fig_2, axis_2 = plt.subplots(figsize=(5,5))
    axis_2.set(frame_on=False)
    axis_2.axis('off')

    table(axis_2, tfpn_df, 
                colWidths = [0.25]*len(tfpn_df.columns),
                rowColours=np.full(len(tfpn_df.index), cl_outcomes['ltgreen']),
                colColours=np.full(len(tfpn_df.columns), cl_outcomes['ltgreen']),
                loc = 'best', edges = 'closed', cellLoc = 'center'
                ).auto_set_column_width(col=list(range(len(tfpn_df.columns))))

    if savePath is not None:
        plt.savefig(os.path.join(savePath, 'results_tfpn.png'), dpi=100)
        plt.close()

    # Make evaluate.json
    evaluate_json = os.path.join(savePath, "evaluate.json")
    with open(evaluate_json, "w") as j:
        json.dump({"iou": iouThreshold, "mAP": mAP, "precison_recall": rpa_df.to_dict('dict')}, j, indent=4)


if __name__ == '__main__':
    main()
