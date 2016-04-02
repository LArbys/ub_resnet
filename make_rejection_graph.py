import os,sys
import ROOT as rt

c = rt.TCanvas("c","c",1400,1400)
c.Draw()

#f = rt.TFile("results/v2_768x768/out_netanalysis_testdata.root","OPEN")
f = rt.TFile("results/ubresnet10/out_netanalysis_testdata.root","OPEN")
hnu = f.Get("hnuprob_nu")
hbg = f.Get("hnuprob_cosmics")
hbg.SetLineColor(rt.kBlack)

nutotal = hnu.Integral()
bgtotal = hbg.Integral()

g = rt.TGraph(hnu.GetXaxis().GetNbins())

cdf_nu = 0.0
cdf_bg = 0.0

spills_per_target_neutrino = 1200.0
swtrig_rejection_factor = 0.03

npts_sr = 0
sr = []
sr2 = []

for ibin in range(0,hnu.GetXaxis().GetNbins()):
    cdf_nu += hnu.GetBinContent(ibin+1)
    cdf_bg += hbg.GetBinContent(ibin+1)
    nu_eff = 1.0-float(cdf_nu)/nutotal
    bg_remain = 1.0-float(cdf_bg)/bgtotal
    g.SetPoint(ibin,nu_eff,bg_remain)
    if nu_eff>0 and bg_remain>0:
        bg_to_nu = (spills_per_target_neutrino/nu_eff)*swtrig_rejection_factor*bg_remain
        sr.append( (nu_eff,1.0/bg_to_nu ) )
        sr2.append( (hnu.GetBinLowEdge(ibin+1),1.0/bg_to_nu) )

gsr = rt.TGraph(len(sr))
gsr2 = rt.TGraph(len(sr2))
for n,pt in enumerate(sr):
    gsr.SetPoint(n,pt[0],pt[1])
for n,pt in enumerate(sr2):
    gsr2.SetPoint(n,pt[0],pt[1])

c.Divide(2,2)
c.cd(1)
hnu.Scale(1.0/nutotal)
hbg.Scale(1.0/bgtotal)
if hbg.GetMaximum()>hnu.GetMaximum():
    hbg.Draw()
    hnu.Draw("same")
else:
    hnu.Draw()
    hbg.Draw("same")

c.cd(2)
g.Draw("ALP")
g.SetTitle(";neutrino efficiency;cosmics remaining")

c.cd(3)
gsr.Draw("ALP")
gsr.SetTitle(";neutrino eff.;est. Nu:Cosmic Ratio")

c.cd(4)
gsr2.Draw("ALP")
gsr2.SetTitle(";neutrino prob. cut;est. Nu:Cosmic Ratio")

c.Update()
#c.SaveAs("test_performance.png")
raw_input()
