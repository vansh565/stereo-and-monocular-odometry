const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let x = 300, y = 300;
let path = [];

// Camera
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

// Send frame
async function sendFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    const dataURL = canvas.toDataURL("image/jpeg");

    const res = await fetch("http://127.0.0.1:5000/process", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ image: dataURL })
    });

    const data = await res.json();

    x += data.x * 5;
    y += data.z * 5;

    path.push([x,y]);

    draw();
}

function draw() {
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,canvas.width,canvas.height);

    ctx.fillStyle = "lime";
    path.forEach(p => ctx.fillRect(p[0], p[1], 2, 2));
}

setInterval(sendFrame, 300);