const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const path = require('path');
const { Canvas, Image, ImageData } = require('canvas');
const faceapi = require('face-api.js');
const fetch = require('node-fetch'); // Import node-fetch

// Initialize face-api.js with the canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Load face-api.js models
const MODEL_URL = path.join(__dirname, 'public', 'models');
let modelsLoaded = false;

const loadModels = async () => {
    if (modelsLoaded) return; // Avoid reloading models if they are already loaded
    try {
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL),
            faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL)
        ]);
        modelsLoaded = true;
        console.log('Face-api models loaded');
    } catch (error) {
        console.error('Error loading models:', error);
        throw error;
    }
};

const app = express();
app.use(bodyParser.json({ limit: '50mb' }));
app.use(express.static('public')); // Serve static files from the 'public' directory

// Ensure directories exist
const ensureDirExists = dir => {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
};

ensureDirExists('./data');
ensureDirExists('./attendance');

// Helper function to convert buffer to image
function bufferToImage(buffer) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = buffer;
    });
}

// Convert Buffer to Base64
function bufferToBase64(buffer) {
    return buffer.toString('base64');
}

// Convert Blob URL to Base64
async function convertBlobUrlToBase64(blobUrl) {
    try {
        const response = await fetch(blobUrl);
        if (!response.ok) {
            throw new Error('Failed to fetch the Blob from the Blob URL');
        }
        const buffer = await response.buffer(); // Convert the Blob URL response to a Buffer
        return bufferToBase64(buffer); // Convert Buffer to Base64 string
    } catch (error) {
        console.error('Error converting Blob URL to Base64:', error);
        throw error;
    }
}

app.post('/helloworld', (req, res) => {
    const { name } = req.body;
    if (!name) {
        return res.status(400).send('Name is required');
    }
    res.send(`Hello, ${name}!`);
});

// Enroll a new user with a Blob URL or Base64 encoded image and a unique username
app.post('/enrollnew', async (req, res) => {
    try {
        await loadModels(); // Ensure models are loaded
        const { username, image } = req.body;
        if (!username || !image) {
            return res.status(400).send('Username and image are required');
        }

        // Handle Blob URL
        let base64Data;
        if (image.startsWith('blob:')) {
            base64Data = await convertBlobUrlToBase64(image);
        } else {
            base64Data = image.replace(/^data:image\/\w+;base64,/, "");
        }

        const imageBuffer = Buffer.from(base64Data, 'base64');
        const imageObj = await bufferToImage(imageBuffer);
        const detections = await faceapi.detectAllFaces(imageObj).withFaceLandmarks().withFaceDescriptors();

        if (detections.length === 0) {
            return res.status(400).send('No face detected in the image');
        }

        const descriptors = detections.map(d => d.descriptor);
        fs.writeFileSync(`./data/${username}.json`, JSON.stringify(descriptors));
        res.sendStatus(200);
    } catch (error) {
        console.error('Error enrolling new user:', error);
        res.status(500).send('Internal Server Error');
    }
});

// Verify a user using a Blob URL or Base64 encoded image
app.post('/verify', async (req, res) => {
    try {
        await loadModels(); // Ensure models are loaded
        const { image } = req.body;
        if (!image) {
            return res.status(400).send('Image is required');
        }

        // Handle Blob URL
        let base64Data;
        if (image.startsWith('blob:')) {
            base64Data = await convertBlobUrlToBase64(image);
        } else {
            base64Data = image.replace(/^data:image\/\w+;base64,/, "");
        }

        const imageBuffer = Buffer.from(base64Data, 'base64');
        const imageObj = await bufferToImage(imageBuffer);
        const detections = await faceapi.detectAllFaces(imageObj).withFaceLandmarks().withFaceDescriptors();

        if (detections.length === 0) {
            return res.status(400).send('No face detected in the image');
        }

        const descriptor = detections[0].descriptor;

        // Log descriptor lengths for debugging
        console.log('Descriptor length for verification:', descriptor.length);

        const employeeFiles = fs.readdirSync('./data');
        const labeledFaceDescriptors = employeeFiles.map(file => {
            const data = JSON.parse(fs.readFileSync(`./data/${file}`));
            // Log descriptor lengths for debugging
            console.log('Descriptor length for file', file, ':', data[0].length);
            return new faceapi.LabeledFaceDescriptors(path.basename(file, '.json'), data.map(d => new Float32Array(d)));
        });

        const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
        const result = faceMatcher.findBestMatch(descriptor);

        if (result.label === 'unknown') {
            return res.status(404).send('No matching user found');
        } else {
            res.json({ username: result.label });
        }
    } catch (error) {
        console.error('Error verifying user:', error);
        res.status(500).send('Internal Server Error');
    }
});

// Save employee descriptors
app.post('/save-employee', (req, res) => {
    try {
        const { employeeId, descriptors } = req.body;
        if (!employeeId || !descriptors) {
            return res.status(400).send('Invalid request data');
        }
        fs.writeFileSync(`./data/${employeeId}.json`, JSON.stringify(descriptors));
        res.sendStatus(200);
    } catch (error) {
        console.error('Error saving employee:', error);
        res.status(500).send('Internal Server Error');
    }
});

// Mark attendance
app.post('/mark-attendance', (req, res) => {
    try {
        const { employeeId } = req.body;
        if (!employeeId) {
            return res.status(400).send('Invalid request data');
        }
        const date = new Date().toISOString().split('T')[0];
        fs.appendFileSync(`./attendance/${date}.txt`, `${employeeId}\n`);
        res.sendStatus(200);
    } catch (error) {
        console.error('Error marking attendance:', error);
        res.status(500).send('Internal Server Error');
    }
});

// Get employee descriptors
app.get('/get-employee-descriptors', (req, res) => {
    try {
        const descriptors = [];
        const employeeFiles = fs.readdirSync('./data');
        employeeFiles.forEach(file => {
            const data = JSON.parse(fs.readFileSync(`./data/${file}`));
            descriptors.push({
                label: path.basename(file, '.json'),
                descriptors: data
            });
        });
        res.json(descriptors);
    } catch (error) {
        console.error('Error getting employee descriptors:', error);
        res.status(500).send('Internal Server Error');
    }
});

const server = app.listen(3001, () => {
    console.log('Server running on port 3001');
});
server.timeout = 600000; // Set timeout to 10 minutes (600000 ms)
