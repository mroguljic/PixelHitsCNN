# coding: utf-8

import os
import sys
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.AlCa.GlobalTag import GlobalTag

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.Modifier_pf_badHcalMitigation_cff import pf_badHcalMitigation

graph_ext = "1dcnn_p1_jun15"
datadir = "/uscms_data/d3/ssekhar/CMSSW_11_1_2/src/TrackerStuff/PixelHitsCNN/data"

# setup minimal options
# options = VarParsing("python")
# options.setDefault("inputFiles", 'root://cms-xrd-global.cern.ch//store/data/Run2018D/SingleMuon/ALCARECO/SiPixelCalSingleMuon-ForPixelALCARECO_UL2018-v1/20000/A28530AF-8EB6-814B-9645-642058515DA2.root')  # noqa
# options.parseArguments()

# define the process to run
process = cms.Process('REC',Run2_2018)
#pf_badHcalMitigation)
# -- Conditions
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("RecoVertex.BeamSpotProducer.BeamSpot_cfi")
#process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
#process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
#process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')

# to get the conditions you need a GT
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# MC
process.GlobalTag = GlobalTag(process.GlobalTag, '105X_upgrade2018_design_v3', '')
# data
#process.GlobalTag = GlobalTag(process.GlobalTag, '112X_dataRun2_v7', '')
# force Generic reco
process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")
process.TTRHBuilderAngleAndTemplate.PixelCPE = cms.string("PixelCPEGeneric")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(5000))
process.source = cms.Source("PoolSource",
  # data
  #fileNames=cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/data/Run2018C/SingleMuon/RAW/v1/000/320/040/00000/407FB3FD-A78E-E811-B816-FA163E120D15.root")
  # MC
 # fileNames=cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIIWinter19PFCalibDR/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/ALCARECO/TkAlMuonIsolated-2018Conditions_ideal_105X_upgrade2018_design_v3-v2/130000/03400616-B7CF-2442-92F2-F0EF0CAD8E6F.root")
fileNames=cms.untracked.vstring("file:MC_10000.root")  
# data
  #fileNames=cms.untracked.vstring("file:52A3B4C3-328E-E811-85D6-FA163E3AB92A.root")
)


# process options
process.options = cms.untracked.PSet(allowUnscheduled=cms.untracked.bool(True),wantSummary=cms.untracked.bool(True))



# setup InferCNN_y by loading the auto-generated cfi (see InferCNN_y.fillDescriptions)
process.load("TrackerStuff.PixelHitsCNN.inferCNN_y_cfi")

process.inferCNN_y = cms.EDAnalyzer('InferCNN_y',
 #graphPath_x = cms.string(os.path.join(datadir, "graph_x_%s.pb"%(graph_ext))),
 graphPath_y = cms.string(os.path.join(datadir, "graph_y_%s.pb"%(graph_ext))),
 #inputTensorName_x = cms.string("input_1"),
 #anglesTensorName_x = cms.string("input_2"),
 inputTensorName_y = cms.string("input_3"),
 anglesTensorName_y = cms.string("input_4"),
 outputTensorName = cms.string("Identity"),
     #mightGet = cms.optional.untracked.vstring,
#     trackCollectionLabel = cms.untracked.InputTag('generalTracks'),
 #    PrimaryVertexCollectionLabel = cms.untracked.InputTag('offlinePrimaryVertices'),
    # rootFileName                 = cms.untracked.string("x_1dcnn.root"),
     #pixelRecHitLabel             = cms.untracked.InputTag('siPixelRecHits')
     )


# define what to run in the path
process.raw2digi_step = cms.Path(process.RawToDigi)   
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction_trackingOnly)
process.endjob_step = cms.EndPath(process.endOfProcess)
#process.siPixelClusters_step = process.siPixelClusters
#process.TrackRefitter_step = cms.Path(
 # process.offlineBeamSpot*
 # process.MeasurementTrackerEvent*
 # process.TrackRefitter
#)
process.pixelCPECNN_step = cms.Path(process.inferCNN_y)

# potentially for the det angle approach
#process.schedule = cms.Schedule(
#  process.raw2digi_step,
#  process.siPixelClusters_step,
#  process.pixelCPECNN_step
#)

# for the track angle approach
process.schedule = cms.Schedule(
  process.raw2digi_step,
#  process.L1Reco_step,
  process.reconstruction_step,
  #process.TrackRefitter_step,
  process.pixelCPECNN_step,
  process.endjob_step
  )
"""
# customisation of the process.
# Automatic addition of the customisation function from Configuration.DataProcessing.RecoTLR
from Configuration.DataProcessing.RecoTLR import customisePostEra_Run2_2018
#call to customisation function customisePostEra_Run2_2018 imported from Configuration.DataProcessing.RecoTLR
process = customisePostEra_Run2_2018(process)
# End of customisation functions
#do not add changes to your config after this point (unless you know what you are doing)
from FWCore.ParameterSet.Utilities import convertToUnscheduled
process=convertToUnscheduled(process)
# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
"""
