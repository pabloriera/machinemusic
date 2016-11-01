var context = new AudioContext();

var BUFFER_SIZE = 4096;
var FRAME_SIZE  = 2048;

var players = [];
var playersIdCounter = 0;



var load_remote_audio = function(url) {
  var request = new XMLHttpRequest();
  request.open('GET', url, true);
  request.responseType = 'arraybuffer';

  request.onload = function() {
      context.decodeAudioData(request.response, function(decodedData) {
        add_player(request.responseURL, decodedData);
      });
  };
  request.send();
};

var load_local_audio = function(localfile) {
      // Create file reader
      var reader = new FileReader();
      
      reader.addEventListener('load', function (e) {
          context.decodeAudioData(reader.result, function(decodedData) {
            add_player(localfile, decodedData);
          });
      });
      reader.addEventListener('error', function () {
          alert('Error reading file');
          console.error('Error reading file');
      });

      reader.readAsArrayBuffer(localfile);

}



var add_player = function(title, decodedData) {
  var id = playersIdCounter++;
  var player = new WAAPlayer(context, decodedData, FRAME_SIZE, BUFFER_SIZE);
  var gain = context.createGain();
  var recorder;

  var ui = new WAAPlayerUI(id, title, player, gain);
  ui.removeCallback = function() {
    player.disconnect();
    gain.disconnect();
    delete players[id];
  }

  players[id] = {
    player : player,
    gain : gain, 
    ui : ui
  };

  player.connect(gain);
  gain.connect(context.destination);
}



