/*
 * Example plugin to demonstrate the direct multi-threaded inference on CNN with TensorFlow 2.
 */

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

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

  std::string inputTensorName_;
  std::string outputTensorName_;

  tensorflow::Session* session_;
};

std::unique_ptr<CacheData> InferCNN::initializeGlobalCache(const edm::ParameterSet& config) {
  // this method is supposed to create, initialize and return a CacheData instance
  CacheData* cacheData = new CacheData();

  // load the graph def and save it
  std::string graphPath = config.getParameter<std::string>("graphPath");
  cacheData->graphDef = tensorflow::loadGraphDef(graphPath);

  // set tensorflow log leven to warning
  tensorflow::setLogging("2");

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
  desc.add<std::string>("graphPath");
  desc.add<std::string>("inputTensorName");
  desc.add<std::string>("outputTensorName");
  descriptions.addWithDefaultLabel(desc);
}

InferCNN::InferCNN(const edm::ParameterSet& config, const CacheData* cacheData)
    : inputTensorName_(config.getParameter<std::string>("inputTensorName")),
      outputTensorName_(config.getParameter<std::string>("outputTensorName")),
      session_(tensorflow::createSession(cacheData->graphDef)) {}

void InferCNN::beginJob() {}

void InferCNN::endJob() {
  // close the session
  tensorflow::closeSession(session_);
}

void InferCNN::analyze(const edm::Event& event, const edm::EventSetup& setup) {
 // define a tensor and fill it with range(10)
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1,15,1});
  for (size_t i = 0; i < 15; i++) {
    if(i == 5) input.tensor<float,3>()(0, i, 0) = 4168.27 ;
    else if (i==6) input.tensor<float,3>()(0, i, 0) = 152646.17;
    else input.tensor<float,3>()(0, i, 0) = 0.;
  }
  // define the output and run
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::run(session_, {{inputTensorName_, input}}, {outputTensorName_}, &outputs);

  // print the output
  std::cout << "THIS IS THE FROM THE CNN -> " << outputs[0].matrix<float>()(0,0) << std::endl << std::endl;
}

DEFINE_FWK_MODULE(InferCNN);
