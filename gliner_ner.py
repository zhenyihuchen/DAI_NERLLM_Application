from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
model.eval()
print("ok")

text = """<p>Mr.  Madgar has been teaching economics part time while working in various professional roles across a wide array of industries.
He taught undergraduate microeconomics and macroeconomics  at Kent State University for 5 years before advancing to teach economic environment &amp; country economic analysis
at the masters’ level at IE, where he has been teaching at now going into his third year. He has 21 years of professional 
work experience ranging from engineering design (where he holds one patent in the US for a ‘nail press’) to director level 
positions throughout supply chain and operations management. Having worked for large Fortune 100 companies such as GE and 
Halliburton to small startups such Millwood Inc., where he currently works,  has an extensive background in applied theory 
in the workplace. A lifelong learner and believer in continuous improvement, he is constantly looking to grow and hone his 
skills, studying new theories and hypotheses in the business setting in an attempt to determine their practical application
under less controlled environments. His passion remains centered around the field of economics where he believes now more than 
ever, a greater understanding and awareness is required as it pertains to tradeoffs and externalities, along with the ‘true costs’
resulting from the decisions that we make and how these impact all those around us and in what manner.
</p><h4>Corporate Experience</h4><p>• Strategic Business Unit Manager, Millwood Inc., USA, 2018-Present</p><p>• 
Global Manufacturing Director, Permasteelisa, USA, 2015-2018</p><p>• Jet Research Center Manager, Halliburton, USA,
2013-2015</p><p>• Corporate Finance Master Six Sigma Black Belt, Cameron ‘a Schlumberger Co’, USA, 2010-2013</p><p>• 
Director of Process Technology, Millwood Inc., USA, 2005-2010</p><p>• Corporate Six Sigma Black Belt, MACtac ‘a Bemis Co’, 
USA, 2003-2005</p><p>• Six Sigma / Lean Manufacturing Black Belt, G.E., USA, 1999-2003</p><h4>Academic Experience
</h4><p>•  Adjunct Professor of Economics, IE Business School, Spain, 2018-Present</p><p>•  Adjunct Level II Professor of Economics, 
Kent State University, USA, 2005-2010</p><h4>Academic Background</h4><p>• E.M.B.A., IE Business School / Brown University, 
Spain / USA, 2018</p><p>• Strategic Decision &amp; Risk Management Certificate, Stanford, USA, 2014</p>
<p>• Executive Certificate in HR Leadership, Cornell, USA, 2013</p><p>• Lean Six Sigma Master Black Belt Certificate, Villanova, 
USA, 2011</p><p>• M.A. Financial Economics, Kent State University, USA, 2004</p><p>• M.B.A., Youngstown State University, 
USA, 2001</p><p>• B.E. Electrical Engineering, Youngstown State University, USA 2000</p>"""


labels = [
    "educational institution where studied",
    "location of educational institution", 
    "academic degree or qualification earned",
    "years of study or graduation year",
    "employer company or organization",
    "workplace location or company headquarters",
    "subject or course taught by professor",
    "academic program or degree level taught",
    "teaching institution or university where professor works"
]



entities = model.predict_entities(text, labels, threshold = 0.4)

for entity in entities:
    print(entity["text"], "->", entity["label"])


# 3 comolumns:
#     - academic background: studies & universities
#     - academic experience: courses (subject + uni + (program))
#     - corporate experience / professional experience: companies and locations
    
#     - ver cuantas rows no tienen heading - solution = data annotations
    
    
    
    
    
    
    
# labels for academic experience column =  "subject or course taught by professor", "academic program or degree level taught", "teaching institution or university where professor works"
# labels for academic background column = "educational institution where studied", "location of educational institution", "academic degree or qualification earned",  "years of study or graduation year"
# labels for professional experience column = "employer company or organization", "workplace location or company headquarters"
