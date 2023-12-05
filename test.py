from transformers import pipeline

question_answerer = pipeline("question-answering")
summarizer = pipeline("summarization")
generator = pipeline("text-generation")
unmasker = pipeline("fill-mask")

context = """

Strasbourg is the prefecture and largest city of the Grand Est region of eastern France and the official seat of the European Parliament. Located at the border with Germany in the historic region of Alsace, it is the prefecture of the Bas-Rhin department.

In 2020, the city proper had 290,576 inhabitants and both the Eurométropole de Strasbourg (Greater Strasbourg) and the Arrondissement of Strasbourg had 511,552 inhabitants.[8] Strasbourg's metropolitan area had a population of 853,110 in 2019,[4] making it the eighth-largest metro area in France and home to 14 per cent of the Grand Est region's inhabitants. The transnational Eurodistrict Strasbourg-Ortenau had a population of roughly 1,000,000 in 2022. Strasbourg is one of the de facto four main capitals of the European Union (alongside Brussels, Luxembourg and Frankfurt), as it is the seat of several European institutions, such as the European Parliament, the Eurocorps and the European Ombudsman of the European Union. An organization separate from the European Union, the Council of Europe (with its European Court of Human Rights, its European Directorate for the Quality of Medicines most commonly known in French as "Pharmacopée Européenne", and its European Audiovisual Observatory) is also located in the city.

Together with Basel (Bank for International Settlements), Geneva (United Nations), The Hague (International Court of Justice) and New York City (United Nations world headquarters), Strasbourg is among the few cities in the world that is not a state capital that hosts international organisations of the first order.[9] The city is the seat of many non-European international institutions such as the Central Commission for Navigation on the Rhine and the International Institute of Human Rights.[10] It is the second city in France in terms of international congress and symposia, after Paris. Strasbourg's historic city centre, the Grande Île (Grand Island), was classified a World Heritage Site by UNESCO in 1988, with the newer "Neustadt" being added to the site in 2017.[11] Strasbourg is immersed in Franco-German culture and although violently disputed throughout history, has been a cultural bridge between France and Germany for centuries, especially through the University of Strasbourg, currently the second-largest in France, and the coexistence of Catholic and Protestant culture. It is also home to the largest Islamic place of worship in France, the Strasbourg Grand Mosque.

Economically, Strasbourg is an important centre of manufacturing and engineering, as well as a hub of road, rail, and river transportation. The port of Strasbourg is the second-largest on the Rhine after Duisburg in Germany, and the second-largest river port in France after Paris.
"""
question = "What is the size of the port of Strasbourg ?"

answer = question_answerer(question=question, context=context)
summarize = summarizer(question)
generation = generator(answer['answer']+ " "+summarize[0]['summary_text'])


print (generation)