# coding: utf-8

import os
import sys
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.AlCa.GlobalTag import GlobalTag
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

# -- Conditions
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

graph_ext = "1dcnn_p1_apr12"

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
# options = VarParsing("python")
# options.setDefault("inputFiles", 'root://cms-xrd-global.cern.ch//store/data/Run2018D/SingleMuon/ALCARECO/SiPixelCalSingleMuon-ForPixelALCARECO_UL2018-v1/20000/A28530AF-8EB6-814B-9645-642058515DA2.root')  # noqa
# options.parseArguments()

# define the process to run
process = cms.Process("TEST")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(-1))
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/data/Run2018C/SingleMuon/RAW/v1/000/320/065/00000/8C070B38-338E-E811-A4D1-FA163E781D28.root")

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(True),
)

# to get the conditions you need a GT
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag = GlobalTag(process.GlobalTag, '112X_dataRun2_v7', '')
                            
process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")
process.TTRHBuilderAngleAndTemplate.PixelCPE = cms.string("PixelCPEGeneric")

# setup InferCNN by loading the auto-generated cfi (see InferCNN.fillDescriptions)
#process.load("TrackerStuff.PixelHitsCNN.inferCNN_cfi
# CLEANUP 
process.inferCNN = cms.EDAnalyzer('InferCNN',
     graphPath_x = cms.string(os.path.join(datadir, "graph_x_%s.pb"%(graph_ext))),
     graphPath_y = cms.string(os.path.join(datadir, "graph_y_%s.pb"%(graph_ext))),
     inputTensorName_x = cms.string("input_1"),
     anglesTensorName_x = cms.string("input_2"),
     inputTensorName_y = cms.string("input_3"),
     anglesTensorName_y = cms.string("input_4"),
     outputTensorName = cms.string("Identity"),
     #mightGet = cms.optional.untracked.vstring,
    # trackCollectionLabel = cms.untracked.InputTag('generalTracks'),
    # PrimaryVertexCollectionLabel = cms.untracked.InputTag('offlinePrimaryVertices'),
     #pixelRecHitLabel             = cms.untracked.InputTag('siPixelRecHits')
   )


# define what to run in the path
process.raw2digi_step = cms.Path(process.RawToDigi)   
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
#process.siPixelClusters_step = process.siPixelClusters
#process.TrackRefitter_step = cms.Path(
 # process.offlineBeamSpot*
 # process.MeasurementTrackerEvent*
 # process.TrackRefitter
#)
process.pixelCPECNN_step = cms.Path(process.inferCNN)

# potentially for the det angle approach
#process.schedule = cms.Schedule(
#  process.raw2digi_step,
#  process.siPixelClusters_step,
#  process.pixelCPECNN_step
#)

# for the track angle approach
process.schedule = cms.Schedule(
  process.raw2digi_step,
  process.L1Reco_step,
  process.reconstruction_step,
  #process.TrackRefitter_step,
  process.pixelCPECNN_step
)
