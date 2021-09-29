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

class forMorris : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
public:
	explicit forMorris(const edm::ParameterSet&, const CacheData*);
	~forMorris(){};

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
	static const int MAXCLUSTER = 50000;
	static const int SIMHITPERCLMAX = 10;             // max number of simhits associated with a cluster/rechit
	//float fClSimHitLx[MAXCLUSTER][SIMHITPERCLMAX];    // X local position of simhit 
	//float fClSimHitLy[MAXCLUSTER][SIMHITPERCLMAX];
	//float x_gen[MAXCLUSTER], x_nn[MAXCLUSTER]; 
	float fClSimHitLx[SIMHITPERCLMAX];    // X local position of simhit 
	float fClSimHitLy[SIMHITPERCLMAX];
	float x_gen, x_nn;
	float eta[MAXCLUSTER], phi[MAXCLUSTER]; int layer[MAXCLUSTER], n_double[MAXCLUSTER]; 
	float eta_all[MAXCLUSTER], phi_all[MAXCLUSTER]; int layer_all[MAXCLUSTER];
	int count=0, total_count = 0, n_L1 = 0, n_L2 = 0, n_L3 = 0, n_L4 = 0, n_end = 0,idx=-1; char path[100], infile1[300], infile2[300], infile3[300], infile4[300];

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
	//float clsize_1[MAXCLUSTER][2], clsize_2[MAXCLUSTER][2], clsize_3[MAXCLUSTER][2], clsize_4[MAXCLUSTER][2], clsize_5[MAXCLUSTER][2], clsize_6[MAXCLUSTER][2];
	struct timeval now0, now1;
    struct timezone timz;


	};

	std::unique_ptr<CacheData> forMorris::initializeGlobalCache(const edm::ParameterSet& config) 
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

	void forMorris::globalEndJob(const CacheData* cacheData) {
	// reset the graphDef
		if (cacheData->graphDef != nullptr) {
			delete cacheData->graphDef;
		}

	}

	void forMorris::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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

	forMorris::forMorris(const edm::ParameterSet& config, const CacheData* cacheData)
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
			phi[i]=9999.0;
			eta[i]=9999.0;
			layer[i]=-999;
			n_double[i]=-999;
			
			
		}
		sprintf(path,"TrackerStuff/PixelHitsCNN/txt_files");

		sprintf(infile2,"%s/forMorris_allclusters.txt",path);
		gen_file = fopen(infile2, "w");
		
		sprintf(infile3,"%s/forMorris_doublepix.txt",path);
		nn_file = fopen(infile3, "w");

		
	}

	void forMorris::beginJob() {

	}

	void forMorris::endJob() {
	// close the session
		tensorflow::closeSession(session_x);

		//fclose(nn_file);
		//fclose(sim_file);

	}

	void forMorris::analyze(const edm::Event& event, const edm::EventSetup& setup) {


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
//		TrackerHitAssociator *associate(0);

	//	associate = new TrackerHitAssociator(event,trackerHitAssociatorConfig_);

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

//		float clusbuf[TXSIZE][TYSIZE], clusbuf_x_temp[TXSIZE], clusbuf_x[TXSIZE];

		static int ix,iy;
		int prev_count = count;
		int prev_total_count = total_count;
		//int id = count-1;
		for (auto const& track : *tracks) {

			//id++;
			//layer[count] = id;

			auto etatk = track.eta();
			auto phitk = track.phi();

			auto const& trajParams = track.extra()->trajParams();
			assert(trajParams.size() == track.recHitsSize());
			auto hb = track.recHitsBegin();

			for (unsigned int h = 0; h < track.recHitsSize(); h++) {
				
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

				

				
				
			

			auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
			if (!pixhit)
				continue;


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
				  //   int minPixelRow(); // Minimum pixel layer in the x direction (low edge).
				  //   int maxPixelRow(); // Maximum pixel layer in the x direction (top edge).
				  //   int minPixelCol(); // Minimum pixel layer in the y direction (left edge).
				  //   int maxPixelCol(); // Maximum pixel layer in the y direction (right edge).
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
			int n_double_x = 0, n_double_y = 0;

			int double_row = -1, double_col = -1;
			int irow_sum = 0, icol_sum = 0;
			for (int i = 0; i < cluster.size(); ++i) {
				auto pix = cluster.pixel(i);
				int irow = int(pix.x) - row_offset;
				int icol = int(pix.y) - col_offset;
					//double pixels skip
				if ((int)pix.x == 79 || (int)pix.x == 80){
				if(irow!=double_row){	
				 n_double_x++; 
				double_row=irow;}
//				printf("irow = %i, pix.adc = %f\n",irow,float(pix.adc));} 
				}
				if ((int)pix.y % 52 == 0 || (int)pix.y % 52 == 51 ){
				if(icol!=double_col){ 
				n_double_y++; 
				double_col = icol;}
				}
				irow_sum+=irow;
				icol_sum+=icol;
				if(float(pix.adc) > cluster_max) cluster_max = float(pix.adc); 
				//if(float(pix.adc) < cluster_min) cluster_min = float(pix.adc); 

			}
			//printf("max = %f, min = %f\n",cluster_max,cluster_min);
			int clustersize_x = cluster.sizeX(), clustersize_y = cluster.sizeY();
			mid_x = round(float(irow_sum)/float(cluster.size()));
			mid_y = round(float(icol_sum)/float(cluster.size()));
			int offset_x = 6 - mid_x;
			int offset_y = 10 - mid_y;

			if(n_double_x>0 || n_double_y>0){
			if (subdetid != PixelSubdetector::PixelBarrel) layer[count] = 9; 
            else layer[count] = tkTpl.pxbLayer(hit_detId);
			eta[count] = etatk;
			phi[count] = phitk;
			n_double[count] = n_double_x+n_double_y;
            count++;
    		}
		switch(tkTpl.pxbLayer(hit_detId)){
		case 1: n_L1++; break;
		case 2: n_L2++; break;
		case 3: n_L3++; break;
		case 4: n_L4++; break;
		default: n_end++;
		}
			layer_all[total_count] = tkTpl.pxbLayer(hit_detId);
			eta_all[total_count] = etatk;
			phi_all[total_count] = phitk;
    		total_count++;
        }
    }

    printf("double width count = %i\n",count);
    printf("total count = %i\n",total_count);
    printf("n_L1 = %i, n_L2 = %i, n_L3 = %i, n_L4 = %i, n_end = %i\n",n_L1,n_L2,n_L3,n_L4,n_end);
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
    	fprintf(nn_file,"%i %f %f %i\n", layer[i],eta[i],phi[i],n_double[i]);

    }
    for(int i=prev_total_count;i<total_count;i++){
    	fprintf(gen_file,"%i %f %f\n", layer[i],eta[i],phi[i]);
    	
    }
    

}
DEFINE_FWK_MODULE(forMorris);
