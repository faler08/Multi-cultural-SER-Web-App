# Multi-cultural-SER-Web-App
An user-friendly web application part of my thesis on Speech Emotion Recognition. It can detect Fear, Happiness, Sadness, Anger, Disgust and Neutralism.

## How is it done?

It uses the best model trained for my project, whose code can be found here: https://github.com/faler08/Multi-cultural-SER.

The web app is a Flask application that allows any user to interact directly with the Whisper & Classification Head model. In short, it is a model that combines the embeddings resulting from Whisper's encoder layers with a logistic classification layer. For more info, check the memoir: https://drive.google.com/file/d/1A4KY3_MyAC-UHrad4UOQ_kZ_XKO0jtUR/view?usp=sharing.

To provide the service in a established URL, we use [ngrok](https://ngrok.com/), which allows us to allocate our PC port in a fixed URL that they lend us. Yet, the model runs in our own server, meaning that we have to keep our machine up and running for this to work.
