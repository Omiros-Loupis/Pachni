"""Χτίζει το index.html από το src/template.html και τα δεδομένα του data/."""
import json, os
ROOT=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(ROOT,'data')
dem=[int(x) for x in open(os.path.join(DATA,'raw_dem.txt')).read().strip().split(',')]
cells=[]
for c in open(os.path.join(DATA,'raw_era.txt')).read().strip().split(';'):
    p=c.split('|')
    cells.append({'lat':float(p[0]),'lon':float(p[1]),'elev':int(p[2]),'g19':int(p[3]),'g20':int(p[4]),'h':[int(x) for x in p[5].split()]})
years=list(range(2000,2026))
H={}
for part in open(os.path.join(DATA,'raw_hist.txt')).read().strip().split(';'):
    y,v=part.split(':'); H[y]=[int(x) for x in v.split(',')]
if H:
    years=[y for y in years if str(y) in H]
    hist=[[H[str(y)][i] for y in years] for i in range(49)]
else:
    hist=None
st=[
 {'name':'Αριδαίας','short':'Αριδαία','lat':40.9833,'lon':22.0538,'elev':120,'t':-5.1,'day':'20/3'},
 {'name':'Κερασιάς Πέλλας','short':'Κερασιά','lat':40.8766,'lon':21.9624,'elev':690,'t':-4.3,'day':'20/3'},
 {'name':'Καρυώτισσας (Γιαννιτσά)','short':'Καρυώτισσα','lat':40.7667,'lon':22.3167,'elev':35,'t':-1.1,'day':'19/3'},
 {'name':'Νάουσας (σε ταράτσα)','short':'Νάουσα','lat':40.6249,'lon':22.0701,'elev':365,'t':1.2,'day':'19/3'},
 {'name':'Βέροιας (πόλη, σε ταράτσα)','short':'Βέροια','lat':40.5222,'lon':22.2050,'elev':158,'t':1.0,'day':'19/3'},
]
towns=[{'name':'Έδεσσα','lat':40.802,'lon':22.047},{'name':'Σκύδρα','lat':40.767,'lon':22.155},{'name':'Γιαννιτσά','lat':40.792,'lon':22.408},{'name':'Αλεξάνδρεια','lat':40.627,'lon':22.447,'left':True}]
test=[['Αριδαία','Κλειστή λεκάνη',-5.1,-2.8,-0.8],['Πλατύκαμπος (Λάρισα)','Κάμπος',-2.3,2.6,2.1],['Αγιά (Λάρισα)','Κάμπος, πρόποδες',-1.4,1.6,3.2],['Καρυώτισσα (Γιαννιτσά)','Ανοιχτός κάμπος',-1.1,-1.7,-0.8],['Νάουσα','Πλαγιά, σε ταράτσα',1.2,-2.9,-0.9]]
D={'dem':{'lat0':40.45,'lon0':21.85,'n':36,'step':0.02,'e':dem},'cells':cells,'hist':hist,'years':years,'st':st,'towns':towns,'test':test}
html=open(os.path.join(ROOT,'src','template.html')).read().replace('__DATA__', json.dumps(D,ensure_ascii=False,separators=(',',':')))
open(os.path.join(ROOT,'index.html'),'w').write(html)
print('index.html:', len(html), 'bytes,', len(years), 'years of history')
