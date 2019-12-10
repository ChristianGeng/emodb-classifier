
# Table of Contents

1.  [Short Version of the Steps](#org74f3dbd)
2.  [Step 1 - Building the dataset](#org56644e6)
3.  [Step 2 - Build Classifier](#org7a2c9c0)
4.  [Step 3 - Generate Report](#org78a9e25)
5.  [Deployment Docker - Unfinished](#org94ea296)

Document the Steps that were needed to carry out the audEERING coding Task


<a id="org74f3dbd"></a>

# Short Version of the Steps

Allgemein: Cookiecutter Data Science Template

Step 4 - Docker Deployment

Unfinished. Bisher ist lediglich das image gebaut. 


<a id="org56644e6"></a>

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


<a id="org7a2c9c0"></a>

# Step 2 - Build Classifier

-   Nur die Functionals wurden Extrahiert, also nicht mehrere Analyseframes per Äusserung

    ./gradlew trainModel 

-   train<sub>model.py</sub> code ist ziemlich ad hoc. Idealerweise will der irgendwann mit sklearn
    BaseEstimator kompatibel werden.
-   Voice Activity Detection? Verzerrungen durch silence
-   Modelle:  GMM ist nicht wirklich fertig. Die Ergebnnisse sind noch instabil; keine llds
-   Interpretation bisher nicht durchgeführt


<a id="org78a9e25"></a>

# Step 3 - Generate Report

Run the model for Report Generation

    ./gradlew makeReportGMMBasic
    ./gradlew makeReportsvmBasic

Eine Variante ist implemtiert, s. src/models/make<sub>report.py</sub>. 
Diese nimmt die papermill bibliothek her und lässt 'report<sub>nb.ipynb</sub>' parametrisiert (model name) laufen. Der
Output landet dann in einem intermediate notebook. Dieses wird dann mit nbconvert konvertiert (tmp.html)

-   Notice: bug in papermill + ipykernel: Läuft trotzdem

-   Alternative sacredboard -mu  mongodb://root:example@localhost:27017 mnist

Mongo admin interface; localhost:8081
Sacred: <https://pypi.org/project/sacred/>
 sacredboard -mu  mongodb://root:example@localhost:27017 torch-bilstm-jean-christophe
 <http://localhost:8081/>
Show decorators:
 /D/myfiles/2019/Sacred-MNIST/train<sub>convnet.py</sub>

-   Accuracy: Normalisieren


<a id="org94ea296"></a>

# Deployment Docker - Unfinished

    docker build -t training_image .
    Successfully built a58fc48440db
    Successfully tagged training_image:latest

