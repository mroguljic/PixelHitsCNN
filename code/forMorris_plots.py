import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.backends.backend_pdf import PdfPages
from scipy import optimize

import ROOT
from ROOT import *

gStyle.SetOptStat(1)
gROOT.SetBatch(1)
gStyle.SetOptFit(1);


# double width count = 32967
# total count = 240077
n = [50966,43488,36097,28673] 
n_end = 80853

double_info = np.genfromtext("txt_files/forMorris_doublepix.txt")
layer = double_info[:,0]
eta = double_info[:,1]
phi = double_info[:,2]
n_double = double_info[:,3]


print(eta.min(),eta.max())
print(phi.min(),phi.max())

for l in [1,2,3,4]:
	
	res = ROOT.TH1F("Fraction of double width pixels vs track eta - BPIX %i"%l,"Fraction of double width pixels vs track eta - BPIX %i"%l,50,eta.min(),eta.max())
	res.SetDirectory(0)
	for entry,weight in zip(eta[layer==l],n_double[layer==l]):
		res.Fill(entry,weight)

	res.Scale(n[l])
	canvas = ROOT.TCanvas (" canvas ")
	canvas.cd()
	res.SetMarkerColor(ROOT.kRed);
    res.SetMarkerStyle(20)
	res.SetMarkerSize(0.6)
	res.SetLineColor(ROOT.kRed)
	res.GetXaxis().SetTitle("eta")
	res.GetYaxis().SetTitle("Fraction of double width pixels")
	res.SetTitle("Fraction of double width pixels vs track eta - BPIX %i"%l)
	res.Draw("pe")
	canvas.Print("plots/forMorris/dpix_eta_%s.png"%l)

	res2 = ROOT.TH1F("Fraction of double width pixels vs track phi - BPIX %i"%l,"Fraction of double width pixels vs track phi - BPIX %i"%l,50,phi.min(),phi.max())
	res2.SetDirectory(0)
	for entry,weight in zip(phi[layer==l],n_double[layer==l]):
		res2.Fill(entry,weight)

	res2.Scale(n[l])
	canvas = ROOT.TCanvas (" canvas ")
	canvas.cd()
	res2.SetMarkerColor(ROOT.kRed);
    res2.SetMarkerStyle(20)
	res2.SetMarkerSize(0.6)
	res2.SetLineColor(ROOT.kRed)
	res2.GetXaxis().SetTitle("phi")
	res2.GetYaxis().SetTitle("Fraction of double width pixels")
	res2.SetTitle("Fraction of double width pixels vs track phi - BPIX %i"%l)
	res2.Draw("pe")
	canvas.Print("plots/forMorris/dpix_phi_%s.png"%l)