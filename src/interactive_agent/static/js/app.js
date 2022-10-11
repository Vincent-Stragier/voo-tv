$(document).ready(function () {
  var socket = io.connect();

  socket.on("speech", function (msg) {
    console.log(
      "Received speech command :: text: '" +
        msg.text +
        "', voice: " +
        msg.voice +
        ", rate: " +
        msg.rate +
        ", volume: " +
        msg.volume
    );

    var audio = new Audio(
      "/a?text=" +
        msg.text +
        "&voice=" +
        msg.voice +
        "&rate=" +
        msg.rate +
        "&volume=" +
        msg.volume
    );
    audio.loop = false;
    audio.play();
  });
});
