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

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"


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

class InferNN_y : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
public:
	explicit InferNN_y(const edm::ParameterSet&, const CacheData*);
	~InferNN_y(){};

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
	tensorflow::Session* session_y;
	bool use_det_angles;
	std::string cpe;
	bool use_generic, use_generic_detangles;
	TFile *fFile; TTree *fTree;
	static const int MAXCLUSTER = 80000;
	static const int SIMHITPERCLMAX = 10;             // max number of simhits associated with a cluster/rechit
	float fClSimHitLx[SIMHITPERCLMAX];    // X local position of simhit 
	float fClSimHitLy[SIMHITPERCLMAX];
	float y_nn,y_gen;
	float dy_gen[MAXCLUSTER], dy_nn[MAXCLUSTER]; int index[MAXCLUSTER]; 
	int idx=-1,count=0,double_count=0,doubledouble_count=0; char path[100], infile1[300], infile2[300], infile3[300], infile4[300];
	edm::InputTag fTrackCollectionLabel, fPrimaryVertexCollectionLabel;
	
	edm::EDGetTokenT<std::vector<reco::Track>> TrackToken;
	edm::EDGetTokenT<reco::VertexCollection> VertexCollectionToken;
	edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> TrackerTopoToken;

	edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> PixelDigiSimLinkToken;
	edm::EDGetTokenT<edm::SimTrackContainer> SimTrackContainerToken;
	edm::EDGetTokenT<edm::SimVertexContainer> SimVertexContainerToken;

	FILE *nn_file, *gen_file, *sim_file, *clustersize_y_file;
	TrackerHitAssociator::Config trackerHitAssociatorConfig_;
	float micronsToCm = 1e-4;
	float pixelsize_x = 100., pixelsize_y = 150., pixelsize_z = 285.0;
	int mid_x = 0, mid_y = 0;
	float clsize_1[MAXCLUSTER][2], clsize_2[MAXCLUSTER][2], clsize_3[MAXCLUSTER][2], clsize_4[MAXCLUSTER][2], clsize_5[MAXCLUSTER][2], clsize_6[MAXCLUSTER][2];
	

	};

	std::unique_ptr<CacheData> InferNN_y::initializeGlobalCache(const edm::ParameterSet& config) 
	{

	// this method is supposed to create, initialize and return a CacheData instance
		CacheData* cacheData = new CacheData();

	// load the graph def and save it
		std::string graphPath_y = config.getParameter<std::string>("graphPath_y");
		cacheData->graphDef = tensorflow::loadGraphDef(graphPath_y);

	// set tensorflow log leven to warning
		tensorflow::setLogging("2");
	//init();

		return std::unique_ptr<CacheData>(cacheData);
	}

	void InferNN_y::globalEndJob(const CacheData* cacheData) {
	// reset the graphDef
		if (cacheData->graphDef != nullptr) {
			delete cacheData->graphDef;
		}

	}

	void InferNN_y::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	// defining this function will lead to a *_cfi file being generated when compiling
		edm::ParameterSetDescription desc;
		desc.add<std::string>("graphPath_y");
		desc.add<std::string>("inputTensorName_y");
		desc.add<std::string>("anglesTensorName_y");
		desc.add<std::string>("outputTensorName");
		desc.add<bool>("use_det_angles");
		desc.add<std::string>("cpe");
		desc.add<bool>("use_generic");
		desc.add<bool>("use_generic_detangles");
		desc.add<bool>("associatePixel");
		desc.add<bool>("associateStrip");
		desc.add<bool>("associateRecoTracks");
		desc.add<std::string>("pixelSimLinkSrc");
		desc.add<std::string>("stripSimLinkSrc");
		desc.add<std::vector<std::string>>("ROUList");
		descriptions.addWithDefaultLabel(desc);
	}

	InferNN_y::InferNN_y(const edm::ParameterSet& config, const CacheData* cacheData)
	: inputTensorName_y(config.getParameter<std::string>("inputTensorName_y")),
	anglesTensorName_y(config.getParameter<std::string>("anglesTensorName_y")),
	outputTensorName_(config.getParameter<std::string>("outputTensorName")),
	session_y(tensorflow::createSession(cacheData->graphDef)),
	use_det_angles(config.getParameter<bool>("use_det_angles")),
	cpe(config.getParameter<std::string>("cpe")),
	use_generic(config.getParameter<bool>("use_generic")),
	use_generic_detangles(config.getParameter<bool>("use_generic_detangles")),
	fTrackCollectionLabel(config.getUntrackedParameter<InputTag>("trackCollectionLabel", edm::InputTag("generalTracks"))),
	fPrimaryVertexCollectionLabel(config.getUntrackedParameter<InputTag>("PrimaryVertexCollectionLabel", edm::InputTag("offlinePrimaryVertices"))),
	trackerHitAssociatorConfig_(config, consumesCollector()) {

		TrackToken              = consumes <std::vector<reco::Track>>(fTrackCollectionLabel) ;
		VertexCollectionToken   = consumes <reco::VertexCollection>(fPrimaryVertexCollectionLabel) ;
		TrackerTopoToken        = esConsumes <TrackerTopology, TrackerTopologyRcd>();

		PixelDigiSimLinkToken   = consumes <edm::DetSetVector<PixelDigiSimLink>>(edm::InputTag("simSiPixelDigis")); 
		SimTrackContainerToken  = consumes <edm::SimTrackContainer>(edm::InputTag("g4SimHits")); 
		SimVertexContainerToken = consumes <edm::SimVertexContainer>(edm::InputTag("g4SimHits")); 
		count = 0;

	//initializations
		for(int i=0;i<MAXCLUSTER;i++){
			dy_nn[i]=9999.0;
			dy_gen[i]=9999.0;
			index[i]=-999;
			
			//for(int j=0;j<SIMHITPERCLMAX;j++){
			//	fClSimHitLx[i][j]=-999.0;
			//	fClSimHitLy[i][j]=-999.0;
			//}
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
		if(use_generic && !use_generic_detangles){
		sprintf(infile1,"%s/generic_MC_y.txt",path);
		gen_file = fopen(infile1, "w");
		}
		else if(use_generic && use_generic_detangles){
		sprintf(infile1,"%s/generic_MC_y_detangles.txt",path);
		gen_file = fopen(infile1, "w");
		}
		else if(!use_generic && !use_generic_detangles){
		sprintf(infile1,"%s/template_MC_y.txt",path);
		gen_file = fopen(infile1, "w");
		}
		else {
		printf("USING TEMPLATE WITH DETANGLES IS WRONG\n");
		return;
		}

		//sprintf(infile2,"%s/simhits_MC_x.txt",path);
		//sim_file = fopen(infile2, "w");

		if(use_det_angles){
		sprintf(infile3,"%s/%s_MC_y_detangles.txt",path,cpe.c_str());
		nn_file = fopen(infile3, "w");

		sprintf(infile4,"%s/%s_MC_perclustersize_y_detangles.txt",path,cpe.c_str());
		clustersize_y_file = fopen(infile4, "w");
		}
		else{
		sprintf(infile3,"%s/%s_MC_y.txt",path,cpe.c_str());
		nn_file = fopen(infile3, "w");

		sprintf(infile4,"%s/%s_MC_perclustersize_y.txt",path,cpe.c_str());
		clustersize_y_file = fopen(infile4, "w");
		}

		
	}

	void InferNN_y::beginJob() {

	}

	void InferNN_y::endJob() {
	// close the session
		tensorflow::closeSession(session_y);

		//fclose(nn_file);
		//fclose(sim_file);

	}

	void InferNN_y::analyze(const edm::Event& event, const edm::EventSetup& setup) {


		//if (sim_file==NULL) {
		//	printf("couldn't open simhit output file/n");
		//	return ;
		//}
	
		if (nn_file==NULL) {
			printf("couldn't open residual output file/n");
			return ;
		}
		edm::ESHandle<TrackerTopology> tTopoHandle = setup.getHandle(TrackerTopoToken);
		auto const& tkTpl = *tTopoHandle;
		// get geometry
	
		std::vector<PSimHit> vec_simhits_assoc;
		TrackerHitAssociator *associate(0);

		associate = new TrackerHitAssociator(event,trackerHitAssociatorConfig_);

	//get the map
		edm::Handle<reco::TrackCollection> tracks;

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

		float clusbuf[TXSIZE][TYSIZE], clusbuf_y[TYSIZE],clusbuf_y_temp[TYSIZE];

		static int ix,iy;
		int prev_count = count;
		//int id = count-1;
		for (auto const& track : *tracks) {

			

			auto etatk = track.eta();

			auto const& trajParams = track.extra()->trajParams();
			assert(trajParams.size() == track.recHitsSize());
			auto hb = track.recHitsBegin();

			for (unsigned int h = 0; h < track.recHitsSize(); h++) {

				idx++;
				index[count] = idx;
				auto hit = *(hb + h);
				if (!hit->isValid()){
				//	printf("hit is not valid\n");
					continue;
				}
				if (hit->geographicalId().det() != DetId::Tracker) {
            		continue; 
         		 }
				auto id = hit->geographicalId();
				DetId hit_detId = hit->geographicalId();
			// check that we are in the pixel detector
				auto subdetid = (id.subdetId());

				if (subdetid != PixelSubdetector::PixelBarrel){ //&& subdetid != PixelSubdetector::PixelEndcap)
			//		printf("not barrel\n");
					continue;
				}
			if (tkTpl.pxbLayer(hit_detId) != 1){ //only L1
			//	printf("not L1\n");
				continue;
			}

			auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
			if (!pixhit){
			//	printf("hit is not vali\n");
				continue;
			}


			//some initialization
			for(int j=0; j<TXSIZE; ++j) {
				for(int i=0; i<TYSIZE; ++i) {
				clusbuf[j][i] = 0.;
				clusbuf_y_temp[i] = 0.;
				clusbuf_y[i] = 0.;
				} 
				//clusbuf_x[j] = 0.;
			} 
			for(int i=0;i<SIMHITPERCLMAX;i++){
				fClSimHitLx[i] = -999.0;
				fClSimHitLy[i] = -999.0;
			}

			// get the cluster
				auto clustp = pixhit->cluster();
			if (clustp.isNull())
				continue;
			auto const& cluster = *clustp;
			const std::vector<SiPixelCluster::Pixel> pixelsVec = cluster.pixels();

			auto const& ltp = trajParams[h];

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
	         //printf("cluster.minPixelRow() = %i\n",cluster.minPixelRow());
	         //printf("cluster.minPixelCol() = %i\n",cluster.minPixelCol());
			// Store the coordinates of the center of the (0,0) pixel of the array that
			// gets passed to PixelTempReco1D
			// Will add these values to the output of  PixelTempReco1D
			float tmp_x = float(row_offset) + 0.5f;
			float tmp_y = float(col_offset) + 0.5f;

			float cotAlpha=ltp.dxdz();
			float cotBeta=ltp.dydz();
			//https://github.com/cms-sw/cmssw/blob/master/RecoLocalTracker/SiPixelRecHits/src/PixelCPEBase.cc#L263-L272
			LocalPoint trk_lp = ltp.position();
			float trk_lp_x = trk_lp.x();
			float trk_lp_y = trk_lp.y();

			Topology::LocalTrackPred loc_trk_pred =Topology::LocalTrackPred(trk_lp_x, trk_lp_y, cotAlpha, cotBeta);
			LocalPoint lp; 
			auto geomdetunit = dynamic_cast<const PixelGeomDetUnit*>(pixhit->detUnit());
			if(!geomdetunit) continue;
			auto const& topol = geomdetunit->specificTopology();
			lp = topol.localPosition(MeasurementPoint(tmp_x, tmp_y), loc_trk_pred);
			if(use_det_angles) lp = topol.localPosition(MeasurementPoint(tmp_x, tmp_y));

			//printf("%f %f \n",cotAlpha,cotBeta);
			if(use_det_angles){
				auto const& theOrigin = geomdetunit->surface().toLocal(GlobalPoint(0, 0, 0));
				LocalPoint lp2 = topol.localPosition(
					MeasurementPoint(cluster.x(), cluster.y()));
				auto gvx = lp2.x() - theOrigin.x();
				auto gvy = lp2.y() - theOrigin.y();
				auto gvz = -1.f /theOrigin.z();	
					// calculate angles
				cotAlpha = gvx * gvz;
				cotBeta = gvy * gvz;
				//printf("detangles: %f %f \n",cotAlpha,cotBeta);
			}

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
			float cluster_max = 0.;

			int n_double = 0, n_double_y = 0;
			int clustersize=0;
			int double_col[5]; for(int i=0;i<5;i++)double_col[i]=-1;
			int irow_sum = 0, icol_sum = 0;
			for (int i = 0; i < cluster.size(); ++i) {
				auto pix = cluster.pixel(i);
				int irow = int(pix.x) - row_offset;
				int icol = int(pix.y) - col_offset;
				if ((irow >= mrow) || (icol >= mcol)) continue;
				if ((int)pix.x == 79 || (int)pix.x == 80){ 
				}
				if ((int)pix.y % 52 == 0 || (int)pix.y % 52 == 51){
				int flag=0;
				for(int j=0;j<5;j++){
				 	if(icol==double_col[j]) {flag = 1; break;}
				}
				if(flag!=1) {double_col[n_double]=icol; n_double++;}
				}
				irow_sum+=irow;
				icol_sum+=icol;
				clustersize++;
				if(float(pix.adc) > cluster_max) cluster_max = float(pix.adc); 
				//if(float(pix.adc) < cluster_min) cluster_min = float(pix.adc); 

			}
			if(n_double>0){
			//printf("MORE THAN 2 DOUBLE COL in Y = %i, SKIPPING\n",n_double);
			continue; //currently can only deal with single double pix
			}
			if(clustersize==0) {printf("EMPTY CLUSTER, SKIPPING\n");continue;}
			int k=0;
			int clustersize_x = cluster.sizeX(), clustersize_y = cluster.sizeY();
			mid_x = round(float(irow_sum)/float(clustersize));
			mid_y = round(float(icol_sum)/float(clustersize));
			int offset_x = 6 - mid_x;
			int offset_y = 10 - mid_y;

			//double_col = 0;
  // Copy clust's pixels (calibrated in electrons) into clusMatrix;
			for (int i = 0; i < cluster.size(); ++i) {
				auto pix = cluster.pixel(i);
				int irow = int(pix.x) - row_offset + offset_x;
				int icol = int(pix.y) - col_offset + offset_y;
					//printf("irow = %i, icol = %i\n",irow,icol);
					//printf("mrow = %i, mcol = %i\n",mrow,mcol);

				if ((irow >= mrow+offset_x) || (icol >= mcol+offset_y)){
				printf("irow or icol exceeded, SKIPPING\n");
				 continue;
				}
				if ((int)pix.y % 52 == 0 || (int)pix.y % 52 == 51 ){
					int flag=0;
                     for(int j=0;j<5;j++){
                             if(icol==double_col[j]) {flag = 1; break;}
                     }
                     if(flag!=1) {double_col[k]=icol; k++;}
                     }
				
				//normalized value
				//if(cluster_max!=cluster_min)
				//clusbuf[irow][icol] = (float(pix.adc))/cluster_max;
				clusbuf[irow][icol] = (float(pix.adc));
				//else clusbuf[irow][icol] = 1.;
 				    //printf("pix[%i].adc = %i, pix.x = %i, pix.y = %i, irow = %i, icol = %i\n",i,pix.adc,pix.x,pix.y,(int(pix.x) - row_offset),int(pix.y) - col_offset);

			}
		
			for(int i = 0;i < TYSIZE; i++){
				for(int j = 0; j < TXSIZE; j++){
					clusbuf_y_temp[i] += clusbuf[j][i];
				}
				//if(clusbuf_y[i] > cluster_max) cluster_max = clusbuf_y[i]; 
				//if(clusbuf_y[i] < cluster_min) cluster_min = clusbuf_y[i] ;
			}
			if(k==1 && clustersize_y>20) {printf("clustersize_y = %i > 20, SKIPPING\n", clustersize_y);continue;}
			if(k==2 && clustersize_x>19) {printf("clustersize_y = %i > 19, SKIPPING\n", clustersize_y);continue;}
			if(k==1) double_count++;
			if(k==2) doubledouble_count++;
			int j = 0;
			//convert double pixels to single - ONLY WORKS FOR 1D
			/*
			for (int i = 0; i < cluster.size(); ++i) {
				auto pix = cluster.pixel(i);
				int irow = int(pix.x) - row_offset + offset_x;
				printf("irow = %i, pix.adc = %f\n",irow,float(pix.adc));
				if ((int)pix.x == 79 || (int)pix.x == 80){
					clusbuf_x[irow] = clusbuf_x_temp[j]/2.;
					clusbuf_x[irow+1] = clusbuf_x_temp[j]/2.;
					offset_x++;

				}
				else clusbuf_x[irow] = clusbuf_x_temp[j];

				j++;
			}
			*/
			for(int i = 0;i < TYSIZE; i++){
               if(((k==1 || k==2) && i==double_col[0] && clustersize_y>1)||(k==2 && i==double_col[1])){
                printf("TREATING A DOUBLE WIDTH PIXEL\n");	
		clusbuf_y[i] = clusbuf_y_temp[j]/2.;
					clusbuf_y[i+1] = clusbuf_y_temp[j]/2.;
					i++;
					if(k==2) double_col[1]++;
                }
                else clusbuf_y[i] = clusbuf_y_temp[j];
		j++;
            }
	    /*
            if(k==2){
	            j=TYSIZE-1;
	            for(int i=0;i<TYSIZE;i++){
			clusbuf_y_temp[i] = clusbuf_y[i];
			clusbuf_y[i]=0.;
			}
	            for(int i = TYSIZE-1;i >=0; i--){
	                if(i==double_col[1] && clustersize_y>1){
					printf("TREATING second DOUBLE WIDTH PIX\n");
	                	clusbuf_y[i] = clusbuf_y_temp[j]/2.;
						clusbuf_y[i-1] = clusbuf_y_temp[j]/2.;
						i--;
	                }
	                else clusbuf_y[i] = clusbuf_y_temp[j];
					j--;
	            }
        	}
		*/
            //compute cluster max
			cluster_max = 0.;
			for(int i = 0;i < TYSIZE; i++){
			if(clusbuf_y[i] > cluster_max) cluster_max = clusbuf_y[i] ; 
			}
			//normalize 1d inputs
			for(int i = 0; i < TYSIZE; i++) clusbuf_y[i] = clusbuf_y[i]/cluster_max;

				//===============================
				// define a tensor and fill it with cluster projection
			tensorflow::Tensor cluster_flat_y(tensorflow::DT_FLOAT, {1,TYSIZE,1});
			tensorflow::Tensor cluster_(tensorflow::DT_FLOAT, {1,TXSIZE,TYSIZE,1});
    		// angles
			tensorflow::Tensor angles(tensorflow::DT_FLOAT, {1,2});
			angles.tensor<float,2>()(0, 0) = cotAlpha;
			angles.tensor<float,2>()(0, 1) = cotBeta;

			for (int i = 0; i < TYSIZE; i++) {
				cluster_flat_y.tensor<float,3>()(0, i, 0) = 0;
				for (int j = 0; j < TXSIZE; j++){
            //1D projection in x
					cluster_flat_y.tensor<float,3>()(0, i, 0) = clusbuf_y[i];
					cluster_.tensor<float,4>()(0, j, i, 0) = clusbuf[j][i];
					
					//printf("%i ",int(clusbuf[i][j]));

				}
					//printf("\n");
			}
			
				// define the output and run
			std::vector<tensorflow::Tensor> output_y;
			//if(cpe=="cnn2d") tensorflow::run(session_y, {{inputTensorName_y,cluster_}, {anglesTensorName_y,angles}}, {outputTensorName_}, &output_y);
			//else tensorflow::run(session_y, {{inputTensorName_y,cluster_flat_y}, {anglesTensorName_y,angles}}, {outputTensorName_}, &output_y);
				// convert microns to cms
			//y_nn = output_y[0].matrix<float>()(0,0);
				//printf("x = %f\n",y_nn[count]);
			y_nn=0;
			y_nn = (y_nn+pixelsize_y*(mid_y))*micronsToCm; 

			//	printf("cota = %f, cotb = %f, y_nn = %f\n",cotAlpha,cotBeta,y_nn[count]);
				// go back to module coordinate system
			y_nn+=lp.y(); 
				// get the generic position
			y_gen = pixhit->localPosition().y();
				//get sim hits
			vec_simhits_assoc.clear();
			vec_simhits_assoc = associate->associateHit(*pixhit);

			int iSimHit = 0;

			for (std::vector<PSimHit>::const_iterator m = vec_simhits_assoc.begin(); 
				m < vec_simhits_assoc.end() && iSimHit < SIMHITPERCLMAX; ++m) 
			{



				fClSimHitLx[iSimHit]    = ( m->entryPoint().x() + m->exitPoint().x() ) / 2.0;
				fClSimHitLy[iSimHit]    = ( m->entryPoint().y() + m->exitPoint().y() ) / 2.0;

				++iSimHit;

            } // end sim hit loop
			if(iSimHit==0){ 
		   		printf("iSimHit = 0 for count = %i\n",count);	
				return;
			}
			for(int i = 0;i<SIMHITPERCLMAX;i++){

				if(fabs(y_nn-fClSimHitLy[i])<fabs(dy_nn[count]))
				dy_nn[count] = y_nn - fClSimHitLy[i];
				
				if(fabs(y_gen-fClSimHitLy[i])<fabs(dy_gen[count]))
				dy_gen[count] = y_gen - fClSimHitLy[i];
			}	
			if(dy_gen[count] >= 999.0 || dy_nn[count] >= 999.0){
				printf("ERROR: Residual is %f %f >= 999.0\n",dy_gen[count],dy_nn[count]);
				return;
			}
			//if(abs(dy_nn[count]*1e4)>100){
			//printf("residual > 100. no of double pixels = %i at cols = %i,%i\n",k,double_col[0],double_col[1]);
			//printf("dy_nn = %f, dy_gen = %f\n",dy_nn[count]*1e4,dy_gen[count]*1e4);
			
			  //for(int j=0;j<TYSIZE;j++) printf("%f ",clusbuf_y[j]);
			//printf("\n");
			
			//}
		//	printf("Generic position: %f\n ",(y_gen[count]-lp.x())*1e4);
		//	printf("nn position: %f\n ",(y_nn[count]-lp.x())*1e4);
		//	printf("simhit_x =");
	
//			printf("%i\n",count);
            switch(clustersize_y){
            	case 1: 
            	clsize_1[count][0]=y_nn;
            	break;
            	case 2: 
            	clsize_2[count][0]=y_nn;
            	break;
            	case 3: 
            	clsize_3[count][0]=y_nn;
            	break;
            	case 4: 
            	clsize_4[count][0]=y_nn;
            	break;
            	case 5: 
            	clsize_5[count][0]=y_nn;
            	break;
            	case 6: 
            	clsize_6[count][0]=y_nn;
            	break;
            }
            count++;

        }
    }

    printf("cluster count with 1 double width pix = %i\n",double_count);
    printf("cluster count with 2 double width pix = %i\n",doubledouble_count);
    printf("total count = %i\n",count);
    for(int i=prev_count;i<count;i++){
    	//for(int j=0; j<SIMHITPERCLMAX;j++){
    	//	fprintf(sim_file,"%f ", fClSimHitLx[i][j]);
    	//}
    	/*
    	for(int j=0; j<SIMHITPERCLMAX;j++){
    		fprintf(sim_file,"%f ", fClSimHitLy[i][j]);
    	}
    	fprintf(sim_file,"\n");
    	*/
    	fprintf(nn_file,"%i %f\n", index[i],dy_nn[i]);
    	fprintf(gen_file,"%i %f\n", index[i],dy_gen[i]);

    	fprintf(clustersize_y_file,"%f %f %f %f %f %f\n", clsize_1[i][0],clsize_2[i][0],clsize_3[i][0],clsize_4[i][0],clsize_5[i][0],clsize_6[i][0]);
    }
    

}
DEFINE_FWK_MODULE(InferNN_y);
