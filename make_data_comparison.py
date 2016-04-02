import os,sys
import ROOT as rt
from math import sqrt

c = rt.TCanvas("c","c",800,400)
c.Draw()

set = "ubresnet10"

f_test = rt.TFile("results/"+set+"/out_netanalysis_testdata.root","OPEN")
f_data = rt.TFile("results/"+set+"/out_netanalysis_databnb_set1.root","OPEN")
files = {"test":f_test,"data":f_data}
hnu = {}
hbg = {}
nutotal = {}
bgtotal = {}
for name,f in files.items():
    hnu[name] = f.Get("hnuprob_nu")
    hbg[name] = f.Get("hnuprob_cosmics")
    print name,hnu[name],hbg[name]
    hbg[name].SetLineColor(rt.kBlack)
    nutotal[name] = hnu[name].Integral()
    bgtotal[name] = hbg[name].Integral()

# Normalize and set errors
hnu["test"].Scale(1.0/hnu["test"].Integral())
hbg["test"].Scale(1.0/hbg["test"].Integral())
for h in [hnu["data"],hbg["data"]]:
    tot = h.Integral()
    if tot==0:
        continue
    for b in range(0,h.GetNbinsX()):
        bincontent = h.GetBinContent(b+1)
        binerr = sqrt(bincontent)/tot
        bincontent /= tot
        h.SetBinContent( b+1, bincontent )
        h.SetBinError( b+1, binerr )
# scale down neutrinos
#crop_Factor = (768.0*788.0-768*768.0)/(768.0*788.0)
#print "Crop Factor: ",crop_Factor
#hnu["test"].Scale( 1.0/(1200.0*0.03)*(1.0-crop_Factor) )

expected_b2s = (600.0*0.03)/0.5 # number of bg spills per neutrino spill
print "BG 2 SIG: ",expected_b2s
bgfrac = expected_b2s/(expected_b2s+1.0)
sigfrac = 1.0/(expected_b2s+1.0)

hbg["test"].Scale(bgfrac)
hnu["test"].Scale(sigfrac)
hsum = hnu["test"].Clone("hsum")
hsum.Add(hbg["test"])

max = 0
if hbg["test"].GetMaximum()>hnu["test"].GetMaximum():
    hbg["test"].Draw()
    hnu["test"].Draw("same")
else:
    hnu["test"].Draw()
    hbg["test"].Draw("same")
hbg["data"].Draw("sameE1")
hsum.Draw("same")
hsum.SetLineStyle(2)
hsum.SetLineColor(rt.kRed)
c.Update()

raw_input()
