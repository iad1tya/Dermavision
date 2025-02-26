const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error("Error accessing webcam:", err));

document.getElementById('userForm').addEventListener('submit', function (event) {
    event.preventDefault();

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = canvas.toDataURL('image/png');

    const name = document.getElementById('name').value;
    const age = document.getElementById('age').value;

    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, age, image: imageData })
    })
    .then(response => response.json())
    .then(data => alert(`Analysis: ${data.analysis}`))
    .catch(error => console.error("Error:", error));
});
