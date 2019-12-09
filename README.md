
# Table of Contents

1.  [Short Version of the Steps](#orge0617e3)
2.  [Step 1 - Building the dataset](#orga3f1eaa)
3.  [Step 2 - Build Classifier](#orgc032d4b)
4.  [Step 3 - Generate Report](#orga222f8e)
5.  [Deployment Docker - Unfinished](#org5ec7e70)

Document the Steps that were needed to carry out the audEERING coding Task


<a id="orge0617e3"></a>

# Short Version of the Steps

Allgemein: Cookiecutter Data Science Template

Step 1 - Building the dataset

Als Gradle Task implmentiert

Step 2 -  Build Classifier

Benutzt momentan lediglich Functionals und keine llds

Step 3 - Generate Html-Report

Eine Variante ist implemtiert, s. src/models/make<sub>report.py</sub>. 
Diese nimmt die papermill bibliothek her und lässt 'report<sub>nb.ipynb</sub>' parametrisiert (model name) laufen. Der
Output landet dann in einem intermediate notebook. Dieses wird dann mit nbconvert konvertiert (tmp.html)

Step 4 - Docker Deployment

Unfinished. Bisher ist lediglich das image gebaut. 


<a id="orga3f1eaa"></a>

# Step 1 - Building the dataset

Dataset creation is done via gradle task, so for recap, a working gradle installation is
required. The database is expected to have been extracted into the \`data/raw subdirectory\` sich that
the path to the wav-files is \`data/raw/wav/\` 
The task itself can be invoked via

    ./gradlew makeDataSet 

-   wav2features.sh: some env variables are hardcoded.
-   soll man den Datensatz end2end bauen, also inklusive wget? Hätte den Charme, daß man diesen gleich
    gemeinsam mit einem git clone ins image verbauen kann?
-   dann könnte ein daraus abgleiteter Container einen branch pullen und Classifier bauen und report
    gen mitmachen??
-   Directory Input


<a id="orgc032d4b"></a>

# Step 2 - Build Classifier

-   Nur die Functionals wurden Extrahiert, also nicht mehrere Analyseframes per Äusserung

    ./gradlew trainModel 

-   train<sub>model.py</sub> code ist ziemlich ad hoc. Idealerweise will der irgendwann mit sklearn
    BaseEstimator kompatibel werden.
-   Voice Activity Detection? Verzerrungen durch silence
-   Modelle:  GMM ist nicht wirklich fertig. Die Ergebnnisse sind noch instabil; keine llds
-   Interpretation bisher nicht durchgeführt


<a id="orga222f8e"></a>

# Step 3 - Generate Report

Run the model for Report Generation

    ./gradlew makeReportGMMBasic
    ./gradlew makeReportsvmBasic

-   Notice: bug in papermill + ipykernel: Läuft trotzdem

-   Alternative sacredboard -mu  mongodb://root:example@localhost:27017 mnist

Mongo admin interface; localhost:8081
Sacred: <https://pypi.org/project/sacred/>
 sacredboard -mu  mongodb://root:example@localhost:27017 torch-bilstm-jean-christophe
 <http://localhost:8081/>
Show decorators:
 /D/myfiles/2019/Sacred-MNIST/train<sub>convnet.py</sub>

-   Accuracy: Normalisieren


<a id="org5ec7e70"></a>

# Deployment Docker - Unfinished

    docker build -t training_image .
    Successfully built a58fc48440db
    Successfully tagged training_image:latest

    -C /home/christian/bin/opensmile-2.3.0/config/IS13_ComParE.conf  -I /media/win-d/myfiles/2019/emodb-classifier/data/raw/wav/03a01Fa.wav -csvoutput /tmp/results.csv  -appendcsv 1
    -C /home/christian/bin/opensmile-2.3.0/config/IS13_ComParE.conf  -I /media/win-d/myfiles/2019/emodb-classifier/data/raw/wav/03a01Fa.wav -csvoutput /tmp/results.csv  -appendcsv 1
    # Fuer Energy
    SMILExtract -C myconfig/demo1.conf -I ./example-audio/opensmile.wav -O myenergy.csv

