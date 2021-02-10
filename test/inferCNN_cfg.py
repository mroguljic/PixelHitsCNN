# coding: utf-8

import os
import sys
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

graph_ext = "cnn_p1_jan31"

# get the data/ directory
#thisdir = os.path.dirname(os.path.realpath(__file__))
#thisdir = os.getcwd()
#if getattr(sys, 'frozen', False):
#	thisdir = os.path.dirname(sys.executable)
#else:
#        # The application is not frozen
#        # Change this bit to match where you store your data files:
#        thisdir = os.path.dirname(__file__)
#datadir = os.path.join(thisdir,"data")
datadir = "/uscms_data/d3/ssekhar/CMSSW_11_1_2/src/TrackerStuff/PixelHitsCNN/data"

# setup minimal options
options = VarParsing("python")
options.setDefault("inputFiles", 'root://cms-xrd-global.cern.ch//store/data/Run2018D/SingleMuon/ALCARECO/SiPixelCalSingleMuon-ForPixelALCARECO_UL2018-v1/20000/A28530AF-8EB6-814B-9645-642058515DA2.root')  # noqa
options.parseArguments()

# define the process to run
process = cms.Process("TEST")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles))

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(True),
)

# setup InferCNN by loading the auto-generated cfi (see InferCNN.fillDescriptions)
process.load("TrackerStuff.PixelHitsCNN.inferCNN_cfi")
process.inferCNN.graphPath = cms.string(os.path.join(datadir, "graph_x_%s.pb"%(graph_ext)))
#CNN has 2 inputs
process.inferCNN.inputTensorName_1 = cms.string("input_1")
process.inferCNN.inputTensorName_2 = cms.string("input_1") #what is the name?
process.inferCNN.outputTensorName = cms.string("Identity")

# define what to run in the path
process.p = cms.Path(process.inferCNN)
