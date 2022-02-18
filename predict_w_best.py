import spacy

nlp=spacy.load("output/models/model-best")

address_list=["130 W BOSE ST STE 100, PARK RIDGE, IL, 60068, USA",
              "8311 MCDONALD RD, HOUSTON, TX, 77053-4821, USA",
              "PO Box 317, 4100 Hwy 20 E Ste 403, NICEVILLE, FL, 32578-5037, USA",
              "C/O Elon Musk Innovations Inc, 1548 E Florida Avenue, Suite 209, TAMPA, FL, 33613, USA",
              "Seven Edgeway Plaza, C/O Mac Dermott Inc, OAKBROOK TERRACE, IL, 60181, USA"]

for address in address_list:
    doc=nlp(address)
    ent_list=[(ent.text, ent.label_) for ent in doc.ents]
    print("Address string -> "+address)
    print("Parsed address -> "+str(ent_list))
    print("******")
