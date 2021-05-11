/*
 * Example plugin to demonstrate the direct multi-threaded inference on CNN with TensorFlow 2.
 */

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>

#include <TF1.h>
#include "Math/MinimizerOptions.h"
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TObject.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TPostScript.h"
#include "Math/DistFunc.h"
#include "TTree.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
class TObject;
class TTree;
class TH1D;
class TFile;

using namespace std;
using namespace edm;
using namespace reco;
// define the cache object
// it could handle graph loading and destruction on its own,
// but in this example, we define it as a logicless container
struct CacheData {
	CacheData() : graphDef(nullptr) {}
	std::atomic<tensorflow::GraphDef*> graphDef;
};

class InferCNN : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
public:
	explicit InferCNN(const edm::ParameterSet&, const CacheData*);
	~InferCNN(){};

	static void fillDescriptions(edm::ConfigurationDescriptions&);

	// two additional static methods for handling the global cache
	static std::unique_ptr<CacheData> initializeGlobalCache(const edm::ParameterSet&);
	static void globalEndJob(const CacheData*);

private:
	void beginJob();
	void analyze(const edm::Event&, const edm::EventSetup&);
	void endJob();

	std::string inputTensorName_x, inputTensorName_y, anglesTensorName_x, anglesTensorName_y;
	std::string outputTensorName_;
	//std::string     fRootFileName;
	tensorflow::Session* session_x;
	TFile *fFile; TTree *fTree;
	int MAXCLUSTER = 100000;
	float x_gen[MAXCLUSTER], x_1dcnn[MAXCLUSTER], dx[MAXCLUSTER]; 
	int count;
	edm::InputTag fTrackCollectionLabel, fPrimaryVertexCollectionLabel;
	 std::string     fRootFileName;
	edm::EDGetTokenT<std::vector<reco::Track>> TrackToken;
	edm::EDGetTokenT<reco::VertexCollection> VertexCollectionToken;
	//const bool applyVertexCut_;

	//edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
	//edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;
	//edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
	//edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
};

std::unique_ptr<CacheData> InferCNN::initializeGlobalCache(const edm::ParameterSet& config) 
//: applyVertexCut_(config.getUntrackedParameter<bool>("VertexCut", true)){

	{

	// this method is supposed to create, initialize and return a CacheData instance
	CacheData* cacheData = new CacheData();

	// load the graph def and save it
	std::string graphPath_x = config.getParameter<std::string>("graphPath_x");
	cacheData->graphDef = tensorflow::loadGraphDef(graphPath_x);

	// set tensorflow log leven to warning
	tensorflow::setLogging("2");
	//init();

	return std::unique_ptr<CacheData>(cacheData);
}

void InferCNN::globalEndJob(const CacheData* cacheData) {
	// reset the graphDef
	if (cacheData->graphDef != nullptr) {
		delete cacheData->graphDef;
	}
}

void InferCNN::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	// defining this function will lead to a *_cfi file being generated when compiling
	edm::ParameterSetDescription desc;
	desc.add<std::string>("graphPath_x");
	desc.add<std::string>("inputTensorName_x");
	desc.add<std::string>("anglesTensorName_x");
	desc.add<std::string>("outputTensorName");
	descriptions.addWithDefaultLabel(desc);
}

InferCNN::InferCNN(const edm::ParameterSet& config, const CacheData* cacheData)
: inputTensorName_x(config.getParameter<std::string>("inputTensorName_x")),
anglesTensorName_x(config.getParameter<std::string>("anglesTensorName_x")),
outputTensorName_(config.getParameter<std::string>("outputTensorName")),
session_x(tensorflow::createSession(cacheData->graphDef)),
//fVerbose(config.getUntrackedParameter<int>("verbose", 0)),
fTrackCollectionLabel(config.getUntrackedParameter<InputTag>("trackCollectionLabel", edm::InputTag("generalTracks"))),
fPrimaryVertexCollectionLabel(config.getUntrackedParameter<InputTag>("PrimaryVertexCollectionLabel", edm::InputTag("offlinePrimaryVertices"))),
fRootFileName(config.getUntrackedParameter<string>("rootFileName", string("x_1dcnn.root"))) {

		TrackToken              = consumes <std::vector<reco::Track>>(fTrackCollectionLabel) ;
	VertexCollectionToken   = consumes <reco::VertexCollection>(fPrimaryVertexCollectionLabel) ;
	count = 0;

	//initializations
	for(int i=0;i<MAXCLUSTER;i++){
		x_1dcnn[i]=-999.0;
		x_gen[i]=-999.0;
		dx[i]=-999.0;

	}
}

void InferCNN::beginJob() {
printf("IN BEGINJOB");
	fFile = TFile::Open(fRootFileName.c_str(), "RECREATE");
  fFile->cd();
fTree = new TTree("x_rec", "x_rec");
 // fTree->Branch("x_gen",        x_gen,       "x_gen");
  fTree->Branch("x_1dcnn",       x_1dcnn,       "x_1dcnn/F");
fTree->Branch("x_gen",        x_gen,       "x_gen/F");
fTree->Branch("dx_1dcnn",       dx,       "dx_1dcnn/F");
}

void InferCNN::endJob() {
	// close the session
	tensorflow::closeSession(session_x);
//	fTree->Fill();
	fFile->cd();
	  fTree->Write();
	    fFile->Write();
  fFile->Close();
//  delete fFile;
printf("IN ENDJOB");
}

void InferCNN::analyze(const edm::Event& event, const edm::EventSetup& setup) {

		// get geometry
	/*
	edm::ESHandle<TrackerGeometry> tracker = setup.getHandle(trackerGeomToken_);
	assert(tracker.isValid());

	edm::ESHandle<TrackerTopology> tTopoHandle = setup.getHandle(trackerTopoToken_);
	auto const& tkTpl = *tTopoHandle;

	edm::Handle<reco::VertexCollection> vertices;
	if (applyVertexCut_) {
		event.getByToken(offlinePrimaryVerticesToken_, vertices);
		if (!vertices.isValid() || vertices->empty())
			return;
	}
	*/

//TH1F* res_x = new TH1F("h706","dx = x_gen - x_1dcnn (all sig)",120,-300,300);
	
	//get the map
	edm::Handle<reco::TrackCollection> tracks;
	//event.getByToken(TrackToken, tracks);
	int nTk(0);
	
	  try {
    event.getByToken(TrackToken, tracks);
  }catch (cms::Exception &ex) {
//    if (fVerbose > 1) 
cout << "No Track collection with label " << fTrackCollectionLabel << endl;
  }
  if (tracks.isValid()) {
    const std::vector<reco::Track> trackColl = *(tracks.product());
    nTk = trackColl.size();
//    if (fVerbose > 1) 
cout << "--> Track collection size: " << nTk << endl;
  } else {
  //  if (fVerbose > 1)
  cout << "--> No valid track collection" << endl;
  }
  if (!tracks.isValid()) {
		cout << "track collection is not valid" <<endl;
		return;
	}

	printf("Track is valid\n");
//	printf("Track collection size: %d\n",tracks->size());
		//stuff needed for template
	float clusbuf[TXSIZE][TYSIZE];
	int mrow=TXSIZE,mcol=TYSIZE;
//	static float xrec, yrec;
	static int ix,iy;

	for (auto const& track : *tracks) {
		//if (applyVertexCut_ &&
		//	(track.pt() < 0.75 || std::abs(track.dxy((*vertices)[0].position())) > 5 * track.dxyError()))
		//	continue;

//		bool isBpixtrack = false, isFpixtrack = false, crossesPixVol = false;

			// find out whether track crosses pixel fiducial volume (for cosmic tracks)
		auto d0 = track.d0(), dz = track.dz();
//		if (std::abs(d0) < 16 && std::abs(dz) < 50)
//			crossesPixVol = true;

		auto etatk = track.eta();

		auto const& trajParams = track.extra()->trajParams();
		assert(trajParams.size() == track.recHitsSize());
		auto hb = track.recHitsBegin();

		for (unsigned int h = 0; h < track.recHitsSize(); h++) {
			auto hit = *(hb + h);
			if (!hit->isValid())
				continue;
			auto id = hit->geographicalId();

				// check that we are in the pixel
			auto subdetid = (id.subdetId());
//			if (subdetid == PixelSubdetector::PixelBarrel)
//				isBpixtrack = true;
//			if (subdetid == PixelSubdetector::PixelEndcap)
//				isFpixtrack = true;
			if (subdetid != PixelSubdetector::PixelBarrel && subdetid != PixelSubdetector::PixelEndcap)
				continue;
			bool iAmBarrel = subdetid == PixelSubdetector::PixelBarrel;

				// PXB_L4 IS IN THE OTHER WAY
				// CAN BE XORed BUT LETS KEEP THINGS SIMPLE
		//	bool iAmOuter = ((tkTpl.pxbLadder(id) % 2 == 1) && tkTpl.pxbLayer(id) != 4) ||
			//((tkTpl.pxbLadder(id) % 2 != 1) && tkTpl.pxbLayer(id) == 4);

			auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
			if (!pixhit)
				continue;


	//some initialization
			for(int j=0; j<TXSIZE; ++j) {for(int i=0; i<TYSIZE; ++i) {clusbuf[j][i] = 0.;} } 


				// get the cluster
				auto clustp = pixhit->cluster();
			if (clustp.isNull())
				continue;
			auto const& cluster = *clustp;
			const std::vector<SiPixelCluster::Pixel> pixelsVec = cluster.pixels();

			int minPixelRow = 161;
			int minPixelCol = 417;

			for (unsigned int i = 0; i < pixelsVec.size(); ++i) {
					float pixx = pixelsVec[i].x;  // index as float=iteger, row index
					float pixy = pixelsVec[i].y;  // same, col index


					
		//  Find lower left corner pixel and its coordinates
					if((int)pixx < minPixelRow) {
						minPixelRow = (int)pixx; 
					}
					if((int)pixy < minPixelCol) {
						minPixelCol = (int)pixy;
					}
				}  // End loop over pixels


	// Now fill the cluster buffer with charges
				for (unsigned int i = 0; i < pixelsVec.size(); ++i) {

					float pixx = pixelsVec[i].x;  // index as float=iteger, row index
					float pixy = pixelsVec[i].y;  // same, col index
					float pixel_charge = pixelsVec[i].adc;

					ix = (int)pixx - minPixelRow;
					if(ix >= TXSIZE) continue;
					iy = (int)pixy - minPixelCol;
					if(iy >= TYSIZE) continue;

		//skip over double pixels: CHECK THIS

					if ((int)pixx == 79){
						i+=2; continue;
					}
					if ((int)pixy % 52 == 51 ){
						i+=2; continue; 
					}
					clusbuf[ix][iy] = pixel_charge;
				}


				auto const& ltp = trajParams[h];

	//Correct charge with Template1D
				float cotAlpha=ltp.dxdz();
				float cotBeta=ltp.dydz();

	//===============================
	// define a tensor and fill it with cluster projection
  tensorflow::Tensor cluster_flat_x(tensorflow::DT_FLOAT, {1,TXSIZE,1});
    // angles
  tensorflow::Tensor angles(tensorflow::DT_FLOAT, {1,2});
           angles.tensor<float,2>()(0, 0) = cotAlpha;
           angles.tensor<float,2>()(0, 1) = cotBeta;
        //   printf("%s\n","starting x reco");
			for (size_t i = 0; i < TXSIZE; i++) {
            cluster_flat_x.tensor<float,3>()(0, i, 0) = 0;
            for (size_t j = 0; j < TYSIZE; j++){
                //1D projection in x
                cluster_flat_x.tensor<float,3>()(0, i, 0) += clusbuf[i][j];
	
		//printf("%f\n",clusbuf[i][j]);	
            }
          }				
	// define the output and run
				std::vector<tensorflow::Tensor> output_x;
				tensorflow::run(session_x, {{inputTensorName_x,cluster_flat_x}, {anglesTensorName_x,angles}}, {outputTensorName_}, &output_x);
				x_1dcnn[count] = output_x[0].matrix<float>()(0,0);
      //    printf("THIS IS THE FROM THE CNN ->%f\n", xrec);
				
				 x_gen[count] = hit->localPosition().x();
				 dx[count] = x_gen[count] - x_1dcnn[count];
				count++;
				
			}
		}
printf("count = %i\n",count);
fTree->Fill();
	}
	DEFINE_FWK_MODULE(InferCNN);
