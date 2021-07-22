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
#include <iomanip>
#include <sstream>
#include <fstream>

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
#include "Geometry/CommonTopologies/interface/Topology.h"

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"  
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h" 
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"


#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
//#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"


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

class InferCNN_x : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
public:
	explicit InferCNN_x(const edm::ParameterSet&, const CacheData*);
	~InferCNN_x(){};

	static void fillDescriptions(edm::ConfigurationDescriptions&);

	// two additional static methods for handling the global cache
	static std::unique_ptr<CacheData> initializeGlobalCache(const edm::ParameterSet&);
	static void globalEndJob(const CacheData*); // does it have to be static

private:
	void beginJob();
	void analyze(const edm::Event&, const edm::EventSetup&);
	void endJob();

	std::string inputTensorName_x, inputTensorName_y, anglesTensorName_x, anglesTensorName_y;
	std::string outputTensorName_;
	//std::string     fRootFileName;
	tensorflow::Session* session_x;
	TFile *fFile; TTree *fTree;
	static const int MAXCLUSTER = 100000;
		static const int SIMHITPERCLMAX = 10;             // max number of simhits associated with a cluster/rechit
		float fClSimHitLx[MAXCLUSTER][SIMHITPERCLMAX];    // X local position of simhit 
		float fClSimHitLy[MAXCLUSTER][SIMHITPERCLMAX];
		float x_gen[MAXCLUSTER], x_1dcnn[MAXCLUSTER], dx[MAXCLUSTER]; 
		int count; char path[100], infile1[300], infile2[300], infile3[300], infile4[300];
		edm::InputTag fTrackCollectionLabel, fPrimaryVertexCollectionLabel;
		std::string     fRootFileName;
		edm::EDGetTokenT<std::vector<reco::Track>> TrackToken;
		edm::EDGetTokenT<reco::VertexCollection> VertexCollectionToken;
		edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> PixelDigiSimLinkToken;
		edm::EDGetTokenT<edm::SimTrackContainer> SimTrackContainerToken;
		edm::EDGetTokenT<edm::SimVertexContainer> SimVertexContainerToken;
		FILE *cnn_file, *gen_file, *sim_file, *clustersize_x_file;
		TrackerHitAssociator::Config trackerHitAssociatorConfig_;
		float micronsToCm = 1e-4;
		float pixelsize_x = 100., pixelsize_y = 150., pixelsize_z = 285.0;
		int mid_x = 0, mid_y = 0;
		float clsize_1[MAXCLUSTER][2], clsize_2[MAXCLUSTER][2], clsize_3[MAXCLUSTER][2], clsize_4[MAXCLUSTER][2], clsize_5[MAXCLUSTER][2], clsize_6[MAXCLUSTER][2];
		//norm_x = 424455.402838 norm_y = 196253.784684
		float NORM_X = 424455.402838, NORM_Y = 196253.784684;

	//const bool applyVertexCut_;

	//edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
	//edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;
	//edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
	//edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
	};

	std::unique_ptr<CacheData> InferCNN_x::initializeGlobalCache(const edm::ParameterSet& config) 
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

	void InferCNN_x::globalEndJob(const CacheData* cacheData) {
	// reset the graphDef
		if (cacheData->graphDef != nullptr) {
			delete cacheData->graphDef;
		}

	}

	void InferCNN_x::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	// defining this function will lead to a *_cfi file being generated when compiling
		edm::ParameterSetDescription desc;
		desc.add<std::string>("graphPath_x");
		desc.add<std::string>("inputTensorName_x");
		desc.add<std::string>("anglesTensorName_x");
		desc.add<std::string>("outputTensorName");
		desc.add<bool>("associatePixel");
		desc.add<bool>("associateStrip");
		desc.add<bool>("associateRecoTracks");
		desc.add<std::string>("pixelSimLinkSrc");
		desc.add<std::string>("stripSimLinkSrc");
		desc.add<std::vector<std::string>>("ROUList");
		descriptions.addWithDefaultLabel(desc);
	}

	InferCNN_x::InferCNN_x(const edm::ParameterSet& config, const CacheData* cacheData)
	: inputTensorName_x(config.getParameter<std::string>("inputTensorName_x")),
	anglesTensorName_x(config.getParameter<std::string>("anglesTensorName_x")),
	outputTensorName_(config.getParameter<std::string>("outputTensorName")),
	session_x(tensorflow::createSession(cacheData->graphDef)),
//fVerbose(config.getUntrackedParameter<int>("verbose", 0)),
//fTrackCollectionLabel(config.getUntrackedParameter<InputTag>("trackCollectionLabel", edm::InputTag("ALCARECOTkAlMuonIsolated"))),
	fTrackCollectionLabel(config.getUntrackedParameter<InputTag>("trackCollectionLabel", edm::InputTag("generalTracks"))),
	fPrimaryVertexCollectionLabel(config.getUntrackedParameter<InputTag>("PrimaryVertexCollectionLabel", edm::InputTag("offlinePrimaryVertices"))),
	trackerHitAssociatorConfig_(config, consumesCollector()) {

		TrackToken              = consumes <std::vector<reco::Track>>(fTrackCollectionLabel) ;
		VertexCollectionToken   = consumes <reco::VertexCollection>(fPrimaryVertexCollectionLabel) ;
		PixelDigiSimLinkToken   = consumes <edm::DetSetVector<PixelDigiSimLink>>(edm::InputTag("simSiPixelDigis")); 
		SimTrackContainerToken  = consumes <edm::SimTrackContainer>(edm::InputTag("g4SimHits")); 
		SimVertexContainerToken = consumes <edm::SimVertexContainer>(edm::InputTag("g4SimHits")); 
		count = 0;

	//initializations
		for(int i=0;i<MAXCLUSTER;i++){
			x_1dcnn[i]=-999.0;
			x_gen[i]=-999.0;
			dx[i]=-999.0;
			for(int j=0;j<SIMHITPERCLMAX;j++){
				fClSimHitLx[i][j]=-999.0;
				fClSimHitLy[i][j]=-999.0;
			}
			for(int j=0;j<2;j++){
				clsize_1[i][j]=-999.0;
				clsize_2[i][j]=-999.0;
				clsize_3[i][j]=-999.0;
				clsize_4[i][j]=-999.0;			
				clsize_5[i][j]=-999.0;
				clsize_6[i][j]=-999.0;
			}
			
		}
		sprintf(path,"TrackerStuff/PixelHitsCNN/txt_files");

		sprintf(infile1,"generic_MC_x.txt");
	//		gen_file = fopen(infile1, "w");

		sprintf(infile2,"%s/simhits_MC.txt",path);
		sim_file = fopen(infile2, "w");

		sprintf(infile3,"%s/cnn_MC_x.txt",path);
		cnn_file = fopen(infile3, "w");

		sprintf(infile4,"%s/cnn1d_MC_perclustersize_x.txt",path);
		clustersize_x_file = fopen(infile4, "w");

		
	}

	void InferCNN_x::beginJob() {

	}

	void InferCNN_x::endJob() {
	// close the session
		tensorflow::closeSession(session_x);

		fclose(cnn_file);
		fclose(sim_file);

	}

	void InferCNN_x::analyze(const edm::Event& event, const edm::EventSetup& setup) {


		if (sim_file==NULL) {
			printf("couldn't open simhit output file/n");
			return ;
		}
	//	if (cnn_file==NULL) {
	//		printf("couldn't open cnn output file/n");
	//		return ;
	//	}
		if (cnn_file==NULL) {
			printf("couldn't open residual output file/n");
			return ;
		}
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
		std::vector<PSimHit> vec_simhits_assoc;
		TrackerHitAssociator *associate(0);

		associate = new TrackerHitAssociator(event,trackerHitAssociatorConfig_);


	//TH1F* res_x = new TH1F("h706","dx = x_gen - x_1dcnn (all sig)",120,-300,300);

	//get the map
		edm::Handle<reco::TrackCollection> tracks;
	//event.getByToken(TrackToken, tracks);
	//int nTk(0);

		try {
			event.getByToken(TrackToken, tracks);
		}catch (cms::Exception &ex) {
	//if (fVerbose > 1) 
			cout << "No Track collection with label " << fTrackCollectionLabel << endl;
		}
		if (tracks.isValid()) {
			const std::vector<reco::Track> trackColl = *(tracks.product());
		//nTk = trackColl.size();
		//if (fVerbose > 1) 
		//cout << "--> Track collection size: " << nTk << endl;
		} else {
  	//if (fVerbose > 1)
			cout << "--> No valid track collection" << endl;
		}
		if (!tracks.isValid()) {
			cout << "track collection is not valid" <<endl;
			return;
		}

	//printf("Track is valid\n");
	//printf("Track collection size: %d\n",tracks->size());
	//stuff needed for template
		float clusbuf[TXSIZE][TYSIZE];
	//int mrow=TXSIZE,mcol=TYSIZE;
	//static float xrec, yrec;
		static int ix,iy;
		int prev_count = count;
		for (auto const& track : *tracks) {
		//if (applyVertexCut_ &&
		//	(track.pt() < 0.75 || std::abs(track.dxy((*vertices)[0].position())) > 5 * track.dxyError()))
		//	continue;

		//bool isBpixtrack = false, isFpixtrack = false, crossesPixVol = false;

		// find out whether track crosses pixel fiducial volume (for cosmic tracks)
		//auto d0 = track.d0(), dz = track.dz();
		//if (std::abs(d0) < 16 && std::abs(dz) < 50)
		//crossesPixVol = true;

			auto etatk = track.eta();

			auto const& trajParams = track.extra()->trajParams();
			assert(trajParams.size() == track.recHitsSize());
			auto hb = track.recHitsBegin();

			for (unsigned int h = 0; h < track.recHitsSize(); h++) {
				auto hit = *(hb + h);
				if (!hit->isValid())
					continue;
				auto id = hit->geographicalId();

			// check that we are in the pixel detector
				auto subdetid = (id.subdetId());
			//if (subdetid == PixelSubdetector::PixelBarrel)
			//isBpixtrack = true;
			//if (subdetid == PixelSubdetector::PixelEndcap)
			//isFpixtrack = true;
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

				auto const& ltp = trajParams[h];

				float cotAlpha=ltp.dxdz();
				float cotBeta=ltp.dydz();
//=======================================================================================
			/*
			printf("cluster.minPixelRow() = %i\n",cluster.minPixelRow());
			printf("cluster.minPixelCol() = %i\n",cluster.minPixelCol());

			int minPixelRow = 161;
			int minPixelCol = 417;

			float tmp_x = float(minPixelRow) + 0.5f;
 			float tmp_y = float(minPixelCol) + 0.5f;

 			
			printf("pixelsVec.size() = %lu\n",pixelsVec.size());
			// fix issues with the coordinate system: extract the centre pixel and its coords
			for (unsigned int i = 0; i < pixelsVec.size(); ++i) {
					float pixx = pixelsVec[i].x;  // index as float=iteger, row index
					float pixy = pixelsVec[i].y;  // same, col index

					printf("pixelsVec[%i].adc = %i, pixx = %f, pixy = %f\n",i,pixelsVec[i].adc,pixx,pixy);

					
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
					printf("ix = %i, iy = %i\n",ix,iy);
				}
				*/
//========================================================================================
				  // Preparing to retrieve ADC counts from the SiPixeltheClusterParam.theCluster->  In the cluster,
  // we have the following:
  //   int minPixelRow(); // Minimum pixel index in the x direction (low edge).
  //   int maxPixelRow(); // Maximum pixel index in the x direction (top edge).
  //   int minPixelCol(); // Minimum pixel index in the y direction (left edge).
  //   int maxPixelCol(); // Maximum pixel index in the y direction (right edge).
  // So the pixels from minPixelRow() will go into clust_array_2d[0][*],
  // and the pixels from minPixelCol() will go into clust_array_2d[*][0].

				int row_offset = cluster.minPixelRow();
				int col_offset = cluster.minPixelCol();
//			printf("cluster.minPixelRow() = %i\n",cluster.minPixelRow());
//			printf("cluster.minPixelCol() = %i\n",cluster.minPixelCol());
  // Store the coordinates of the center of the (0,0) pixel of the array that
  // gets passed to PixelTempReco1D
  // Will add these values to the output of  PixelTempReco1D
				float tmp_x = float(row_offset) + 0.5f;
				float tmp_y = float(col_offset) + 0.5f;
//			printf("tmp_x = %f, tmp_y = %f\n", tmp_x,tmp_y);

//			printf("cluster.size() = %i\n",cluster.size());

				  // first compute matrix size
				int mrow = 0, mcol = 0;
				for (int i = 0; i != cluster.size(); ++i) {
					auto pix = cluster.pixel(i);
					int irow = int(pix.x);
					int icol = int(pix.y);
					mrow = std::max(mrow, irow);
					mcol = std::max(mcol, icol);
				}
				mrow -= row_offset;
				mrow += 1;
				mrow = std::min(mrow, TXSIZE);
				mcol -= col_offset;
				mcol += 1;
				mcol = std::min(mcol, TYSIZE);
				assert(mrow > 0);
				assert(mcol > 0);
//			printf("mrow = %i, mcol = %i\n",mrow,mcol);
  //float clusbuf[mrow][mcol];
  //memset(clusbuf, 0, sizeof(float) * mrow * mcol);
				int clustersize_x = 0, clustersize_y = 0;
				bool bigPixel=false;
				int irow_sum = 0, icol_sum = 0;
				int same_x = 500, same_y = 500; //random initial value

				for (int i = 0; i < cluster.size(); ++i) {
					auto pix = cluster.pixel(i);
					int irow = int(pix.x) - row_offset;
					int icol = int(pix.y) - col_offset;
					//double pixels skip
					if ((int)pix.x == 79 || (int)pix.x == 80){
						bigPixel=true; break;
					}
					if ((int)pix.y % 52 == 0 || (int)pix.y % 52 == 51 ){
						bigPixel=true; break;
					}
					irow_sum+=irow;
					icol_sum+=icol;
					if(irow != same_x){
						clustersize_x++;
						same_x = irow;
					}
					if(icol != same_y){
						clustersize_y++;
						same_y = icol;
					}
				}
				if(bigPixel) continue;
				//printf("clustersize_x = %i, clustersize_y = %i\n",clustersize_x,clustersize_y);
				mid_x = round(float(irow_sum)/float(cluster.size()));
				mid_y = round(float(icol_sum)/float(cluster.size()));
				int offset_x = 6 - mid_x;
				int offset_y = 10 - mid_y;
				//printf("offset_x = %i, offset_y = %i\n",offset_x,offset_y);
  // Copy clust's pixels (calibrated in electrons) into clusMatrix;
				for (int i = 0; i < cluster.size(); ++i) {
					auto pix = cluster.pixel(i);
					int irow = int(pix.x) - row_offset + offset_x;
					int icol = int(pix.y) - col_offset + offset_y;
					//printf("irow = %i, icol = %i\n",irow,icol);
					//printf("mrow = %i, mcol = %i\n",mrow,mcol);
    // Gavril : what do we do here if the row/column is larger than cluster_matrix_size_x/cluster_matrix_size_y  ?
    // Ignore them for the moment...
					
					if ((irow > mrow+offset_x) || (icol > mcol+offset_y)) continue;
					clusbuf[irow][icol] = float(pix.adc);
 //   printf("pix[%i].adc = %i, pix.x = %i, pix.y = %i, irow = %i, icol = %i\n",i,pix.adc,pix.x,pix.y,irow,icol);

				}
//			printf("fails after filling buffer\n");
 			//https://github.com/cms-sw/cmssw/blob/master/RecoLocalTracker/SiPixelRecHits/src/PixelCPEBase.cc#L263-L272
				LocalPoint trk_lp = ltp.position();
				float trk_lp_x = trk_lp.x();
				float trk_lp_y = trk_lp.y();

				Topology::LocalTrackPred loc_trk_pred =Topology::LocalTrackPred(trk_lp_x, trk_lp_y, cotAlpha, cotBeta);
				LocalPoint lp; 
				auto geomdetunit = dynamic_cast<const PixelGeomDetUnit*>(pixhit->detUnit());
				auto const& topol = geomdetunit->specificTopology();
				lp = topol.localPosition(MeasurementPoint(tmp_x, tmp_y), loc_trk_pred);
		//	printf("fails after lp\n");
				//===============================
				// define a tensor and fill it with cluster projection
				tensorflow::Tensor cluster_flat_x(tensorflow::DT_FLOAT, {1,TXSIZE,1});
    		// angles
				tensorflow::Tensor angles(tensorflow::DT_FLOAT, {1,2});
				angles.tensor<float,2>()(0, 0) = cotAlpha;
				angles.tensor<float,2>()(0, 1) = cotBeta;

				for (size_t i = 0; i < TXSIZE; i++) {
					cluster_flat_x.tensor<float,3>()(0, i, 0) = 0;
					for (size_t j = 0; j < TYSIZE; j++){
            //1D projection in x
						cluster_flat_x.tensor<float,3>()(0, i, 0) += clusbuf[i][j];
					//	cluster_flat_x.tensor<float,3>()(0, i, 0) = 0;
					//	printf("%i ",int(clusbuf[i][j]));

					}
					//printf("\n");
				}
				//for testing purposes:
				/*
				angles.tensor<float,2>()(0, 0) = -0.201142;
				angles.tensor<float,2>()(0, 1) = 6.538521;
				cluster_flat_x.tensor<float,3>()(0, 5, 0) = 0.00982028;
				cluster_flat_x.tensor<float,3>()(0, 6, 0) = 0.3596283;
				*/
				// define the output and run
				std::vector<tensorflow::Tensor> output_x;
				tensorflow::run(session_x, {{inputTensorName_x,cluster_flat_x}, {anglesTensorName_x,angles}}, {outputTensorName_}, &output_x);
				// convert microns to cms
				x_1dcnn[count] = output_x[0].matrix<float>()(0,0);
				//printf("x = %f\n",x_1dcnn[count]);

				x_1dcnn[count] = (x_1dcnn[count]+pixelsize_x*(mid_x))*micronsToCm; 

			//	printf("cota = %f, cotb = %f, x_cnn = %f\n",cotAlpha,cotBeta,x_1dcnn[count]);
				// go back to module coordinate system
				x_1dcnn[count]+=lp.x(); 
				// get the generic position
				x_gen[count] = hit->localPosition().x();
				//get sim hits
				vec_simhits_assoc.clear();
				vec_simhits_assoc = associate->associateHit(*pixhit);

            //fClSimHitN[count] = (int)vec_simhits_assoc.size();
				int iSimHit = 0;

				for (std::vector<PSimHit>::const_iterator m = vec_simhits_assoc.begin(); 
					m < vec_simhits_assoc.end() && iSimHit < SIMHITPERCLMAX; ++m) 
				{



					fClSimHitLx[count][iSimHit]    = ( m->entryPoint().x() + m->exitPoint().x() ) / 2.0;
					fClSimHitLy[count][iSimHit]    = ( m->entryPoint().y() + m->exitPoint().y() ) / 2.0;

					++iSimHit;

            } // end sim hit loop

				// compute the generic residual
				//dx[count] = x_gen[count] - x_1dcnn[count];

//			printf("Generic position: %f\n ",x_gen[count]*1e4);
//			printf("1dcnn position: %f\n ",x_1dcnn[count]*1e4);
//			printf("%i\n",count);
            switch(clustersize_x){
					case 1: 
					clsize_1[count][0]=x_1dcnn[count];
					break;
					case 2: 
					clsize_2[count][0]=x_1dcnn[count];
					break;
					case 3: 
					clsize_3[count][0]=x_1dcnn[count];
					break;
					case 4: 
					clsize_4[count][0]=x_1dcnn[count];
					break;
					case 5: 
					clsize_5[count][0]=x_1dcnn[count];
					break;
					case 6: 
					clsize_6[count][0]=x_1dcnn[count];
					break;
				}
            count++;

        }
    }
//	printf("count = %i\n",count);
		//fTree->Fill();
		//printf("Output from generic:\n");
	//	for(int i=prev_count;i<count;i++){
	//		printf("%f\n", x_gen[i]);
	//	}
	//	printf("Output from 1dcnn:\n");
	//	for(int i=prev_count;i<count;i++){
	//		printf("%f\n", x_1dcnn[i]);
	//	}
	//	printf("dx residual:\n");
	
    for(int i=prev_count;i<count;i++){
    	for(int j=0; j<SIMHITPERCLMAX;j++){
    		fprintf(sim_file,"%f ", fClSimHitLx[i][j]);
    	}
    	for(int j=0; j<SIMHITPERCLMAX;j++){
    		fprintf(sim_file,"%f ", fClSimHitLy[i][j]);
    	}
    	fprintf(sim_file,"\n");
    	fprintf(cnn_file,"%f %f\n", x_gen[i],x_1dcnn[i]);

		fprintf(clustersize_x_file,"%f %f %f %f %f %f\n", clsize_1[i][0],clsize_2[i][0],clsize_3[i][0],clsize_4[i][0],clsize_5[i][0],clsize_6[i][0]);
    }
    

}
DEFINE_FWK_MODULE(InferCNN_x);
