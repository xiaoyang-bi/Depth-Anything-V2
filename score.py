from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="./SoccerNet")
# mySoccerNetDownloader.downloadDataTask(task="depth-2025", split=["train","valid","test","challenge"]) # to access the 2025 challenge part of the dataset
mySoccerNetDownloader.downloadDataTask(task="depth-basketball", split=["train","valid","test"]) # to access the football part of the dataset