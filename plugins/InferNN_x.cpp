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
#include <sys/time.h>

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

class InferNN_x : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
public:
	explicit InferNN_x(const edm::ParameterSet&, const CacheData*);
	~InferNN_x(){};

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
	bool use_det_angles;
	std::string cpe;
	bool use_generic, use_generic_detangles;
	TFile *fFile; TTree *fTree;
	static const int MAXCLUSTER = 80000;
	static const int SIMHITPERCLMAX = 10;             // max number of simhits associated with a cluster/rechit
	//float fClSimHitLx[MAXCLUSTER][SIMHITPERCLMAX];    // X local position of simhit 
	//float fClSimHitLy[MAXCLUSTER][SIMHITPERCLMAX];
	//float x_gen[MAXCLUSTER], x_nn[MAXCLUSTER]; 
	float fClSimHitLx[SIMHITPERCLMAX];    // X local position of simhit 
	float fClSimHitLy[SIMHITPERCLMAX];
	float x_gen, x_nn;
	float dx_gen[MAXCLUSTER], dx_nn[MAXCLUSTER]; int index[MAXCLUSTER]; 
	int count=0, double_count = 0, doubledouble_count = 0,idx=-1; char path[100], infile1[300], infile2[300], infile3[300], infile4[300];
	edm::InputTag fTrackCollectionLabel, fPrimaryVertexCollectionLabel;
	
	edm::EDGetTokenT<std::vector<reco::Track>> TrackToken;
	edm::EDGetTokenT<reco::VertexCollection> VertexCollectionToken;
	edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> TrackerTopoToken;

	edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink>> PixelDigiSimLinkToken;
	edm::EDGetTokenT<edm::SimTrackContainer> SimTrackContainerToken;
	edm::EDGetTokenT<edm::SimVertexContainer> SimVertexContainerToken;

	FILE *nn_file, *gen_file, *sim_file, *clustersize_x_file;
	TrackerHitAssociator::Config trackerHitAssociatorConfig_;
	float micronsToCm = 1e-4;
	float pixelsize_x = 100., pixelsize_y = 150., pixelsize_z = 285.0;
	int mid_x = 0, mid_y = 0;
	float clsize_1[MAXCLUSTER][2], clsize_2[MAXCLUSTER][2], clsize_3[MAXCLUSTER][2], clsize_4[MAXCLUSTER][2], clsize_5[MAXCLUSTER][2], clsize_6[MAXCLUSTER][2];
	struct timeval now0, now1;
	struct timezone timz;


};

std::unique_ptr<CacheData> InferNN_x::initializeGlobalCache(const edm::ParameterSet& config) 
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

void InferNN_x::globalEndJob(const CacheData* cacheData) {
	// reset the graphDef
	printf("in global end job\n");
		//tensorflow::closeSession(session_x);		
	if (cacheData->graphDef != nullptr) {
		delete cacheData->graphDef;
	}

}

void InferNN_x::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	// defining this function will lead to a *_cfi file being generated when compiling
	edm::ParameterSetDescription desc;
	desc.add<std::string>("graphPath_x");
	desc.add<std::string>("inputTensorName_x");
	desc.add<std::string>("anglesTensorName_x");
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

InferNN_x::InferNN_x(const edm::ParameterSet& config, const CacheData* cacheData)
: inputTensorName_x(config.getParameter<std::string>("inputTensorName_x")),
anglesTensorName_x(config.getParameter<std::string>("anglesTensorName_x")),
outputTensorName_(config.getParameter<std::string>("outputTensorName")),
session_x(tensorflow::createSession(cacheData->graphDef)),
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
		dx_nn[i]=9999.0;
		dx_gen[i]=9999.0;
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
		sprintf(infile1,"%s/generic_MC_x.txt",path);
		gen_file = fopen(infile1, "w");
	}
	else if(use_generic && use_generic_detangles){
		sprintf(infile1,"%s/generic_MC_x_detangles.txt",path);
		gen_file = fopen(infile1, "w");
	}
	else if(!use_generic && !use_generic_detangles){
		sprintf(infile1,"%s/template_MC_x.txt",path);
		gen_file = fopen(infile1, "w");
	}
	else {
		printf("USING TEMPLATE WITH DETANGLES IS WRONG\n");
		return;
	}

		//sprintf(infile2,"%s/simhits_MC_x.txt",path);
		//sim_file = fopen(infile2, "w");

	if(use_det_angles){
		sprintf(infile3,"%s/%s_MC_x_detangles.txt",path,cpe.c_str());
		nn_file = fopen(infile3, "w");

		sprintf(infile4,"%s/%s_MC_perclustersize_x_detangles.txt",path,cpe.c_str());
		clustersize_x_file = fopen(infile4, "w");
	}
	else{
		sprintf(infile3,"%s/%s_MC_x.txt",path,cpe.c_str());
		nn_file = fopen(infile3, "w");

		sprintf(infile4,"%s/%s_MC_perclustersize_x.txt",path,cpe.c_str());
		clustersize_x_file = fopen(infile4, "w");
	}


}

void InferNN_x::beginJob() {

}

void InferNN_x::endJob() {
	// close the session
		//tensorflow::closeSession(session_x);
	printf("in end job\n");
		//fclose(nn_file);
		//fclose(sim_file);

}

void InferNN_x::analyze(const edm::Event& event, const edm::EventSetup& setup) {


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

	float clusbuf[TXSIZE][TYSIZE], clusbuf_temp[TXSIZE][TYSIZE], clusbuf_x[TXSIZE];

	static int ix,iy;
	int prev_count = count;
		//int id = count-1;
	for (auto const& track : *tracks) {

		idx++;
		index[count] = idx;

		auto etatk = track.eta();

		auto const& trajParams = track.extra()->trajParams();
		assert(trajParams.size() == track.recHitsSize());
		auto hb = track.recHitsBegin();

		for (unsigned int h = 0; h < track.recHitsSize(); h++) {
		//		idx++;
                //index[count] = idx;

			auto hit = *(hb + h);
			if (!hit->isValid())
				continue;
			if (hit->geographicalId().det() != DetId::Tracker) {
				continue; 
			}
			auto id = hit->geographicalId();
			DetId hit_detId = hit->geographicalId();

			// check that we are in the pixel detector
			auto subdetid = (id.subdetId());



				if (subdetid != PixelSubdetector::PixelBarrel) //&& subdetid != PixelSubdetector::PixelEndcap)
					continue;
				if (tkTpl.pxbLayer(hit_detId) != 1) //only L1
					continue;


				auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
				if (!pixhit)
					continue;


			//some initialization
				for(int j=0; j<TXSIZE; ++j) {
					for(int i=0; i<TYSIZE; ++i) {
						clusbuf[j][i] = 0.;
						clusbuf_temp[j][i] = 0.;
				//clusbuf_y[i] = 0.;
					} 
				//	clusbuf_x_temp[j] = 0.;
					clusbuf_x[j] = 0.;
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
			//printf("%f %f\n",cotAlpha,cotBeta);

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
				//printf("detangles: %f %f\n",cotAlpha,cotBeta);
				}
//			printf("count = %i\n",count);
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
				
				int n_double_x = 0, n_double_y = 0;
				int clustersize = 0;
				int double_row[5], double_col[5]; 
				for(int i=0;i<5;i++){
					double_row[i]=-1;
					double_col[i]=-1;
				}
				
				int irow_sum = 0, icol_sum = 0;
				for (int i = 0; i < cluster.size(); ++i) {
					auto pix = cluster.pixel(i);
					int irow = int(pix.x) - row_offset;
					int icol = int(pix.y) - col_offset;
					if ((irow >= mrow) || (icol >= mcol)) continue;	
					if ((int)pix.x == 79 || (int)pix.x == 80){
						int flag=0;
						for(int j=0;j<5;j++){
							if(irow==double_row[j]) {flag = 1; break;}
						}
						if(flag!=1) {double_row[n_double_x]=irow; n_double_x++;}
					}
					if ((int)pix.y % 52 == 0 || (int)pix.y % 52 == 51){
						int flag=0;
						for(int j=0;j<5;j++){
							if(icol==double_col[j]) {flag = 1; break;}
						}
						if(flag!=1) {double_col[n_double_y]=icol; n_double_y++;}
					}
					irow_sum+=irow;
					icol_sum+=icol;
					clustersize++;
				//if(float(pix.adc) > cluster_max) cluster_max = float(pix.adc); 
				//if(float(pix.adc) < cluster_min) cluster_min = float(pix.adc); 

				}
				if(clustersize==0){printf("EMPTY CLUSTER, SKIPPING\n");continue;}	
				if(n_double_x>2 or n_double_y>2){
		//	printf("MORE THAN 2 DOUBLE COL in X  = %i, SKIPPING\n",n_double);
			continue; //currently can only deal with single double pix
		}
		n_double_x=0; n_double_y=0;
			//printf("max = %f, min = %f\n",cluster_max,cluster_min);
		int clustersize_x = cluster.sizeX(), clustersize_y = cluster.sizeY();
		mid_x = round(float(irow_sum)/float(clustersize));
		mid_y = round(float(icol_sum)/float(clustersize));
		int offset_x = 6 - mid_x;
		int offset_y = 10 - mid_y;
			//printf("mid_x = %i, mid_y = %i, cluster.size = %i, clustersize = %i\n",mid_x,mid_y,cluster.size(),clustersize); 

			//printf("clustersize_x = %i\n",clustersize_x);	
  // Copy clust's pixels (calibrated in electrons) into clusMatrix;
		for (int i = 0; i < cluster.size(); ++i) {
			auto pix = cluster.pixel(i);
			int irow = int(pix.x) - row_offset + offset_x;
			int icol = int(pix.y) - col_offset + offset_y;
					//printf("irow = %i, icol = %i\n",irow,icol);
					//printf("mrow = %i, mcol = %i\n",mrow,mcol);

			if ((irow >= mrow+offset_x) || (icol >= mcol+offset_y)){
				printf("irow or icol exceeded, SKIPPING. irow = %i, mrow = %i, offset_x = %i,icol = %i, mcol = %i, offset_y = %i\n",irow,mrow,offset_x,icol,mcol,offset_y);
				continue;
			}
				//normalized value
				//if(cluster_max!=cluster_min)
				//clusbuf[irow][icol] = (float(pix.adc))/cluster_max;
			if ((int)pix.x == 79 || (int)pix.x == 80){
				int flag=0;
				for(int j=0;j<5;j++){
					if(irow==double_row[j]) {flag = 1; break;}
				}
				if(flag!=1) {double_row[n_double_x]=irow; n_double_x++;}
			}
			if ((int)pix.y % 52 == 0 || (int)pix.y % 52 == 51 ){
				int flag=0;
				for(int j=0;j<5;j++){
					if(icol==double_col[j]) {flag = 1; break;}
				}
				if(flag!=1) {double_col[n_double_y]=icol; n_double_y++;}
			}
			clusbuf_temp[irow][icol] = float(pix.adc);
				//else clusbuf[irow][icol] = 1.;
 				//if(n_double>0) printf("pix[%i].adc = %i, pix.x = %i, pix.y = %i, irow = %i, icol = %i\n",i,pix.adc,pix.x,pix.y,(int(pix.x) - row_offset),int(pix.y) - col_offset);

		}

		
		if(n_double_x==1 && clustersize_x>12) {printf("clustersize_x > 12, SKIPPING\n"); continue;} // NEED TO FIX CLUSTERSIZE COMPUTATION
		if(n_double_x==2 && clustersize_x>11) {printf("clustersize_x > 11, SKIPPING\n"); continue;}
		if(n_double_y==1 && clustersize_y>20) {printf("clustersize_y = %i > 20, SKIPPING\n", clustersize_y);continue;}
		if(n_double_y==2 && clustersize_x>19) {printf("clustersize_y = %i > 19, SKIPPING\n", clustersize_y);continue;}
		/*
		if(n_double_x>0 or n_double_y>0){
			printf("double width cluster of size %i containing %i x double pixels and %i y double pixels\n",clustersize_x,n_double_x,n_double_y);
			for(int i=0;i<TXSIZE;i++){
				for(int f=0;f<TYSIZE;f++)
				printf("%f ",clusbuf_temp[i][f]);
			printf("\n");
			}
		}*/
		//first deal with double width pixels in x
		int k=0,m=0;
		for(int i=0;i<TXSIZE;i++){
			if(i==double_row[m] and clustersize_x>1){
				printf("TREATING DPIX%i IN X\n",m+1);
				for(int j=0;j<TYSIZE;j++){
					clusbuf[i][j]=clusbuf_temp[k][j]/2.;
					clusbuf[i+1][j]=clusbuf_temp[k][j]/2.;
				}
				i++;
				if(m==0 and n_double_x==2) {
					double_row[1]++;
					m++;
				}
			}
			else{
				for(int j=0;j<TYSIZE;j++){
					clusbuf[i][j]=clusbuf_temp[k][j];
				}
			}
			k++;
		}
		k=0;m=0;
		for(int i=0;i<TXSIZE;i++){
			for(int j=0;j<TYSIZE;j++){
				clusbuf_temp[i][j]=clusbuf[i][j];
				clusbuf[i][j]=0.;
			}
		}
		for(int j=0;j<TYSIZE;j++){
			if(j==double_col[m] and clustersize_y>1){
				printf("TREATING DPIX%i IN Y\n",m+1);
				for(int i=0;i<TXSIZE;i++){
					clusbuf[i][j]=clusbuf_temp[i][k]/2.;
					clusbuf[i][j+1]=clusbuf_temp[i][k]/2.;
				}
				j++;
				if(m==0 and n_double_y==2) {
					double_col[1]++;
					m++;
				}
			}
			else{
				for(int i=0;i<TXSIZE;i++){
					clusbuf[i][j]=clusbuf_temp[i][k];
				}
			}
			k++;
		}
		/*
		if(n_double_x>0 or n_double_y>0){
			printf("MODIFIED double width cluster of size %i containing %i x double pixels and %i y double pixels\n",clustersize_x,n_double_x,n_double_y);
			for(int i=0;i<TXSIZE;i++){
				for(int f=0;f<TYSIZE;f++)
				printf("%f ",clusbuf[i][f]);
			printf("\n");
			}
		}*/
			/*
			if(k==1 or k==2){
			printf("double width cluster of size %i containing %i double pixels\n",clustersize_x,k);
			for(int i=0;i<TXSIZE;i++){
				for(int f=0;f<TYSIZE;f++)
				printf("%f ",clusbuf[i][f]);
			printf("\n");
			}
			for(int i = 0;i < TXSIZE; i++){
                printf("%f ",clusbuf_x_temp[i]);
            }
			printf("\n");
			}
			
		int j = 0;
		for(int i = 0;i < TXSIZE; i++){
			if(((k==1 || k==2) && i==double_row[0] && clustersize_x>1)||(k==2 && i==double_row[1])){
				//printf("TREATING first DOUBLE WIDTH PIX\n");
				clusbuf_x[i] = clusbuf_x_temp[j]/2.;
				clusbuf_x[i+1] = clusbuf_x_temp[j]/2.;
				i++;
				if(k==2) double_row[1]++;
			}
			else clusbuf_x[i] = clusbuf_x_temp[j];
			j++;
            }
            
			if(k==1 or k==2){
	         	printf("MODIFIED double width cluster\n");
	         	for(int i = 0;i < TXSIZE; i++){
	          	printf("%f ",clusbuf_x[i]);
	          	}
	         	printf("\n");
     		}
            */
			//compute the 1d projection & compute cluster max
			float cluster_max = 0., cluster_max_2d = 0.;
            for(int i = 0;i < TXSIZE; i++){
			for(int j = 0; j < TYSIZE; j++){
				clusbuf_x[i] += clusbuf[i][j];
				if(clusbuf[i][j]>cluster_max_2d) cluster_max_2d = clusbuf[i][j];
				}
				if(clusbuf_x[i] > cluster_max) cluster_max = clusbuf_x[i] ; 
		    }
			
			//normalize 1d inputs
			for(int i = 0; i < TXSIZE; i++) clusbuf_x[i] = clusbuf_x[i]/cluster_max;

			//normalize 2d inputs
			for(int i = 0;i < TXSIZE; i++){
				for (int j = 0; i < TYSIZE; j++)
				{
					clusbuf[i][j] /= cluster_max_2d;
				}
			}

				//========================================================================================
				// define a tensor and fill it with cluster projection
				tensorflow::Tensor cluster_flat_x(tensorflow::DT_FLOAT, {1,TXSIZE,1});
			tensorflow::Tensor cluster_(tensorflow::DT_FLOAT, {1,TXSIZE,TYSIZE,1});
    		// angles
			tensorflow::Tensor angles(tensorflow::DT_FLOAT, {1,2});
			angles.tensor<float,2>()(0, 0) = cotAlpha;
			angles.tensor<float,2>()(0, 1) = cotBeta;

			for (int i = 0; i < TXSIZE; i++) {
				cluster_flat_x.tensor<float,3>()(0, i, 0) = 0;
				for (int j = 0; j < TYSIZE; j++){
            //1D projection in x
					cluster_flat_x.tensor<float,3>()(0, i, 0) = clusbuf_x[i];
					cluster_.tensor<float,4>()(0, i, j, 0) = clusbuf[i][j];
					
					//printf("%i ",int(clusbuf[i][j]));

				}
					//printf("\n");
			}
			//  Determine current time

    			//gettimeofday(&now0, &timz);
				// define the output and run
			std::vector<tensorflow::Tensor> output_x;
			if(cpe=="cnn2d"){ gettimeofday(&now0, &timz);
				tensorflow::run(session_x, {{inputTensorName_x,cluster_}, {anglesTensorName_x,angles}}, {outputTensorName_}, &output_x);
				gettimeofday(&now1, &timz);
			}
			else {  gettimeofday(&now0, &timz);
				tensorflow::run(session_x, {{inputTensorName_x,cluster_flat_x}, {anglesTensorName_x,angles}}, {outputTensorName_}, &output_x);
				gettimeofday(&now1, &timz);
			}
			// convert microns to cms
			x_nn = output_x[0].matrix<float>()(0,0);

			//printf("x_nn[%i] = %f\n",count,x_nn[count]);
			//if(isnan(x_nn[count])){
			//for(int i=0;i<TXSIZE;i++){
			//	for(int j=0;j<TYSIZE;j++)
			//		printf("%i ",int(clusbuf[i][j]));
			//	printf("\n");
			//}
			//printf("\n");}
			x_nn = (x_nn+pixelsize_x*(mid_x))*micronsToCm; 

				//printf("cota = %f, cotb = %f, x_nn = %f\n",cotAlpha,cotBeta,x_nn[count]);
				// go back to module coordinate system
			x_nn+=lp.x(); 
			//gettimeofday(&now1, &timz);
			float deltaus = now1.tv_usec - now0.tv_usec;
			//printf("elapsed time = %f us\n",deltaus);
				// get the generic position
			x_gen = pixhit->localPosition().x();
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

            	if(fabs(x_nn-fClSimHitLx[i])<fabs(dx_nn[count]))
            		dx_nn[count] = x_nn - fClSimHitLx[i];

            	if(fabs(x_gen-fClSimHitLx[i])<fabs(dx_gen[count]))
            		dx_gen[count] = x_gen - fClSimHitLx[i];
            }	
            if(dx_gen[count] >= 999.0 || dx_nn[count] >= 999.0){
            	printf("ERROR: Residual is dx_gen=%f dx_nn=%f \n",dx_gen[count],dx_nn[count]);
				//for(int i=0;i<TXSIZE;i++){
                                 //for(int f=0;f<TYSIZE;f++)
                                 //printf("%f ",clusbuf[i][f]);
                        // printf("\n");
                         //}
			//	return;
            } 
		//	printf("Generic position: %f\n ",(x_gen[count]-lp.x())*1e4);
		//	printf("nn position: %f\n ",(x_nn[count]-lp.x())*1e4);
		//	printf("simhit_x =");

//			printf("%i\n",count);
            switch(clustersize_x){
            	case 1: 
            	clsize_1[count][0]=x_nn;
            	break;
            	case 2: 
            	clsize_2[count][0]=x_nn;
            	break;
            	case 3: 
            	clsize_3[count][0]=x_nn;
            	break;
            	case 4: 
            	clsize_4[count][0]=x_nn;
            	break;
            	case 5: 
            	clsize_5[count][0]=x_nn;
            	break;
            	case 6: 
            	clsize_6[count][0]=x_nn;
            	break;
            }
            count++;

        }
    }

    printf("cluster count with 1 double width pix = %i\n",double_count);
    printf("cluster count with 2 double width pix = %i\n",doubledouble_count);
    printf("total count = %i\n",count);
    for(int i=prev_count;i<count;i++){
    	/*
    	for(int j=0; j<SIMHITPERCLMAX;j++){
    		fprintf(sim_file,"%f ", fClSimHitLx[i][j]);
    	}
    	//for(int j=0; j<SIMHITPERCLMAX;j++){
    	//	fprintf(sim_file,"%f ", fClSimHitLy[i][j]);
    	//}
    	fprintf(sim_file,"\n");
    	*/
    	fprintf(nn_file,"%i %f\n", index[i],dx_nn[i]);
    	fprintf(gen_file,"%i %f\n",index[i],dx_gen[i]);

    	fprintf(clustersize_x_file,"%f %f %f %f %f %f\n", clsize_1[i][0],clsize_2[i][0],clsize_3[i][0],clsize_4[i][0],clsize_5[i][0],clsize_6[i][0]);
    }
    

}
DEFINE_FWK_MODULE(InferNN_x);
