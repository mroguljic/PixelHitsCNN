# coding: utf-8

import os
import sys
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing
from Configuration.AlCa.GlobalTag import GlobalTag

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.Modifier_pf_badHcalMitigation_cff import pf_badHcalMitigation

n_events = 500
'''
h5_ext = "p1_2018_irrad_BPIXL1"
cpe = "cnn1d"
use_generic = True
use_generic_detangles = False
use_det_angles = True

if(cpe=="cnn1d"): graph_ext = "1dcnn_%s_aug31"%h5_ext
elif(cpe=="cnn2d"): graph_ext = "2dcnn_%s_aug31"%h5_ext
else: graph_ext = "dnn_%s_jul28"%h5_ext

print("n_events = %i, use_generic = %i, use_det_angles = %i, cpe = %s"%(n_events,use_generic,use_det_angles,cpe))

datadir = "/uscms_data/d3/ssekhar/CMSSW_11_1_2/src/TrackerStuff/PixelHitsCNN/data"
'''

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
process.load('Configuration.StandardSequences.Reconstruction_cff')

#process.load('Configuration.StandardSequences.AlCaRecoStreams_cff')

# to get the conditions you need a GT
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
# MC
#process.GlobalTag = GlobalTag(process.GlobalTag, '105X_upgrade2018_design_v3', '') #phase-1 2018 unirradiated
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
# data
#process.GlobalTag = GlobalTag(process.GlobalTag, '112X_dataRun2_v7', '')
#process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
#process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
#process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
#process.load("RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi") 
# force Generic reco
#process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")
#if use_generic: process.TTRHBuilderAngleAndTemplate.PixelCPE = cms.string("PixelCPEGeneric")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(n_events))
process.source = cms.Source("PoolSource",
  # data
  #fileNames=cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/data/Run2018C/SingleMuon/RAW/v1/000/320/040/00000/407FB3FD-A78E-E811-B816-FA163E120D15.root")
  # MC
 # fileNames=cms.untracked.vstring("root://cms-xrd-global.cern.ch//store/mc/RunIIWinter19PFCalibDR/WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8/ALCARECO/TkAlMuonIsolated-2018Conditions_ideal_105X_upgrade2018_design_v3-v2/130000/03400616-B7CF-2442-92F2-F0EF0CAD8E6F.root")
#fileNames=cms.untracked.vstring("file:MC_5000_111X_upgrade2018_realistic_v3.root"),   
#fileNames=cms.untracked.vstring("file:MC_15000_phase1_2018_realistic.root"),
fileNames=cms.untracked.vstring("file:TTbar_13TeV_TuneCUETP8M1_cfi_MC_500_phase1_2018_realistic.root"),
eventsToSkip=cms.untracked.VEventRange('1:442-1:446')
# data
  #fileNames=cms.untracked.vstring("file:52A3B4C3-328E-E811-85D6-FA163E3AB92A.root")
#skipEvents=cms.untracked.uint32(15)

)
print("Using global tag "+process.GlobalTag.globaltag._value)

# process options
process.options = cms.untracked.PSet(allowUnscheduled=cms.untracked.bool(True),wantSummary=cms.untracked.bool(True))

# # # -- Trajectory producer
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.TrackRefitter.src = 'generalTracks'
process.TrackRefitter.NavigationSchool = ""

# # -- RecHit production
process.load("RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi")
process.load("RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi")

#process.load("RecoTracker.TransientTrackingRecHit.TTRHBuilderWithTemplate_cfi")
process.TTRHBuilderAngleAndTemplate.PixelCPE = cms.string("PixelCPEGeneric")


# setup InferNN_x by loading the auto-generated cfi (see InferNN_x.fillDescriptions)
#process.load("TrackerStuff.PixelHitsCNN.inferNN_x_cfi")

process.PixelTree = cms.EDAnalyzer(
        "PixelTree",
        verbose                      = cms.untracked.int32(2),
        rootFileName                 = cms.untracked.string('PixelTree_PU_CH_generic_newIBC.root'),
        phase                        = cms.untracked.int32(1),
        #type                         = cms.untracked.string(getDataset(process.source.fileNames[0])),                                                                                                                                                                          
        globalTag                    = process.GlobalTag.globaltag,
        dumpAllEvents                = cms.untracked.int32(0),
        PrimaryVertexCollectionLabel = cms.untracked.InputTag('offlinePrimaryVertices'),
        muonCollectionLabel          = cms.untracked.InputTag('muons'),
        trajectoryInputLabel         = cms.untracked.InputTag('TrackRefitter::RECO'),
        TTRHBuilder                  = cms.string('WithAngleAndTemplate'),
        trackCollectionLabel         = cms.untracked.InputTag('generalTracks'),
        pixelClusterLabel            = cms.untracked.InputTag('siPixelClusters'),
        pixelRecHitLabel             = cms.untracked.InputTag('siPixelRecHits'),
        HLTProcessName               = cms.untracked.string('HLT'),
        L1GTReadoutRecordLabel       = cms.untracked.InputTag('gtDigis'),
        hltL1GtObjectMap             = cms.untracked.InputTag('hltL1GtObjectMap'),
        HLTResultsLabel              = cms.untracked.InputTag('TriggerResults::HLT'),
        # SimHits                                                                                                                                                                                                                                                               
        accessSimHitInfo             = cms.untracked.bool(True),
        associatePixel               = cms.bool(True),
        associateStrip               = cms.bool(False),
        associateRecoTracks          = cms.bool(False),
        pixelSimLinkSrc              = cms.InputTag("simSiPixelDigis"),
        ROUList                      = cms.vstring(
                'TrackerHitsPixelBarrelLowTof',
                'TrackerHitsPixelBarrelHighTof',
                'TrackerHitsPixelEndcapLowTof',
                'TrackerHitsPixelEndcapHighTof'),
        associateHitbySimTrack       = cms.bool(False),
)

# define what to run in the path
# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)

process.PixelTree_step = cms.Path(process.siPixelRecHits*process.TrackRefitter*process.PixelTree)
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step, process.PixelTree_step)

