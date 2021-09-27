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

double_info = np.genfromtxt("txt_files/forMorris_doublepix.txt")
layer = double_info[:,0]
eta = double_info[:,1]
phi = double_info[:,2]
n_double = double_info[:,3]


print(eta.min(),eta.max())
print(phi.min(),phi.max())

for l in [1,2,3,4]:
	
	res = ROOT.TH1F("double pixels vs eta - BPIX %i"%l,"double pixels vs eta - BPIX %i"%l,50,eta.min(),eta.max())
	res.SetDirectory(0)
	for entry,weight in zip(eta[layer==l],n_double[layer==l]):
		res.Fill(entry,weight)

	res.Scale(1./n[l-1])
	canvas = ROOT.TCanvas (" canvas ","canvas",1700,1000)
	canvas.cd()
	res.SetMarkerColor(ROOT.kRed);
    	res.SetMarkerStyle(20)
	res.SetMarkerSize(0.6)
	res.SetLineColor(ROOT.kRed)
	res.GetXaxis().SetTitle("eta")
	res.GetYaxis().SetTitle("Fraction of double width pixels")
	res.SetTitle("Fraction of double width pixels vs track eta - BPIX %i - Total %.2f%%"%(l,(res.GetEntries()/n[l-1])*100))
	res.Draw("pe")
	canvas.Print("plots/forMorris/dpix_eta_%s.png"%l)

	res2 = ROOT.TH1F("double pixels vs eta - BPIX %i"%l,"double pixels vs eta - BPIX %i"%l,50,phi.min(),phi.max())
	res2.SetDirectory(0)
	for entry,weight in zip(phi[layer==l],n_double[layer==l]):
		res2.Fill(entry,weight)

	res2.Scale(1./n[l-1])
	canvas = ROOT.TCanvas (" canvas ","canvas",1700,1000)
	canvas.cd()
	res2.SetMarkerColor(ROOT.kRed);
    	res2.SetMarkerStyle(20)
	res2.SetMarkerSize(0.6)
	res2.SetLineColor(ROOT.kRed)
	res2.GetXaxis().SetTitle("phi")
	res2.GetYaxis().SetTitle("Fraction of double width pixels")
	res2.SetTitle("Fraction of double width pixels vs track phi - BPIX %i - Total %.2f%%"%(l,(res2.GetEntries()*100)/n[l-1]))
	res2.Draw("pe")
	canvas.Print("plots/forMorris/dpix_phi_%s.png"%l)

res = ROOT.TH1F("double pixels vs eta - FPIX","double pixels vs eta - FPIX",50,eta.min(),eta.max())
res.SetDirectory(0)
for entry,weight in zip(eta[layer==9],n_double[layer==9]):
	res.Fill(entry,weight)

res.Scale(1./n_end)
canvas = ROOT.TCanvas (" canvas ","canvas",1700,1000)
canvas.cd()
res.SetMarkerColor(ROOT.kRed);
res.SetMarkerStyle(20)
res.SetMarkerSize(0.6)
res.SetLineColor(ROOT.kRed)
res.GetXaxis().SetTitle("eta")
res.GetYaxis().SetTitle("Fraction of double width pixels")
res.SetTitle("Fraction of double width pixels vs track eta - FPIX - Total %.2f%%"%((res.GetEntries()*100)/n_end))
res.Draw("pe")
canvas.Print("plots/forMorris/dpix_eta_fpix.png")

res = ROOT.TH1F("double pixels vs phi - FPIX","double pixels vs phi - FPIX",50,phi.min(),phi.max())
res.SetDirectory(0)
for entry,weight in zip(phi[layer==9],n_double[layer==9]):
	res.Fill(entry,weight)

res.Scale(1./n_end)
canvas = ROOT.TCanvas (" canvas ","canvas",1700,1000)
canvas.cd()
res.SetMarkerColor(ROOT.kRed);
res.SetMarkerStyle(20)
res.SetMarkerSize(0.6)
res.SetLineColor(ROOT.kRed)
res.GetXaxis().SetTitle("phi")
res.GetYaxis().SetTitle("Fraction of double width pixels")
res.SetTitle("Fraction of double width pixels vs track phi - FPIX - Total %.2f%%"%((res.GetEntries()*100)/n_end))
res.Draw("pe")
canvas.Print("plots/forMorris/dpix_phi_fpix.png")
