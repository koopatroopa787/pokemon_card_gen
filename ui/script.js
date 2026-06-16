// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const numImagesInput = document.getElementById('numImages');
const numImagesSlider = document.getElementById('numImagesSlider');
const seedInput = document.getElementById('seed');
const randomSeedBtn = document.getElementById('randomSeed');
const generateBtn = document.getElementById('generateBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const gallery = document.getElementById('gallery');
const clearGalleryBtn = document.getElementById('clearGallery');
const modelInfoDiv = document.getElementById('modelInfo');
const modal = document.getElementById('imageModal');
const modalImg = document.getElementById('modalImage');
const closeModal = document.getElementsByClassName('close')[0];
const downloadBtn = document.getElementById('downloadBtn');

// State
let currentModalImage = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadModelInfo();
    syncNumImages();
    setupEventListeners();
});

// Event Listeners
function setupEventListeners() {
    numImagesInput.addEventListener('input', (e) => {
        numImagesSlider.value = e.target.value;
    });

    numImagesSlider.addEventListener('input', (e) => {
        numImagesInput.value = e.target.value;
    });

    randomSeedBtn.addEventListener('click', () => {
        seedInput.value = Math.floor(Math.random() * 1000000);
    });

    generateBtn.addEventListener('click', generateCards);

    clearGalleryBtn.addEventListener('click', () => {
        gallery.innerHTML = '<div class="placeholder"><p>🎲 Click "Generate Cards" to create your Pokemon cards!</p></div>';
    });

    document.getElementById('downloadZipBtn').addEventListener('click', () => downloadBatch('zip'));
    document.getElementById('downloadGridBtn').addEventListener('click', () => downloadBatch('grid'));

    closeModal.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    downloadBtn.addEventListener('click', downloadModalImage);
}

// Sync number of images
function syncNumImages() {
    numImagesSlider.value = numImagesInput.value;
}

// Load Model Info
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/info`);
        const data = await response.json();

        modelInfoDiv.innerHTML = `
            <p><strong>Status:</strong> ${data.model_loaded ? '✅ Loaded' : '❌ Not Loaded'}</p>
            <p><strong>Device:</strong> ${data.device}</p>
            <p><strong>Latent Dim:</strong> ${data.latent_dim}</p>
            <p><strong>Checkpoints:</strong> ${data.available_checkpoints.length}</p>
        `;

        if (!data.model_loaded) {
            generateBtn.disabled = true;
            btnText.textContent = 'Model Not Loaded';
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        modelInfoDiv.innerHTML = `
            <p style="color: var(--error-color);">❌ Cannot connect to API</p>
            <p style="font-size: 0.8rem;">Make sure the server is running</p>
        `;
        generateBtn.disabled = true;
        btnText.textContent = 'API Not Available';
    }
}

// Generate Cards
async function generateCards() {
    const numImages = parseInt(numImagesInput.value);
    const seed = seedInput.value ? parseInt(seedInput.value) : null;

    // Validation
    if (numImages < 1 || numImages > 16) {
        alert('Please select between 1 and 16 cards');
        return;
    }

    // UI Updates
    generateBtn.disabled = true;
    btnText.style.display = 'none';
    btnLoader.style.display = 'block';

    try {
        const requestBody = {
            num_images: numImages
        };

        if (seed !== null) {
            requestBody.seed = seed;
        }

        const response = await fetch(`${API_BASE_URL}/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Generation failed');
        }

        const data = await response.json();

        // Clear placeholder if exists
        const placeholder = gallery.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }

        // Add images to gallery
        data.images.forEach((imageBase64, index) => {
            addImageToGallery(imageBase64, index + 1);
        });

        // Success feedback
        showNotification(`✅ Generated ${numImages} card${numImages > 1 ? 's' : ''} successfully!`);

    } catch (error) {
        console.error('Error generating cards:', error);
        showNotification(`❌ Error: ${error.message}`, true);
    } finally {
        // Reset UI
        generateBtn.disabled = false;
        btnText.style.display = 'block';
        btnLoader.style.display = 'none';
    }
}

// Add Image to Gallery
function addImageToGallery(imageBase64, index) {
    const galleryItem = document.createElement('div');
    galleryItem.className = 'gallery-item';

    const img = document.createElement('img');
    img.src = `data:image/png;base64,${imageBase64}`;
    img.alt = `Pokemon Card ${index}`;

    const overlay = document.createElement('div');
    overlay.className = 'overlay';
    overlay.innerHTML = '<span>🔍</span>';

    galleryItem.appendChild(img);
    galleryItem.appendChild(overlay);

    // Click to enlarge
    galleryItem.addEventListener('click', () => {
        openModal(img.src);
    });

    gallery.appendChild(galleryItem);
}

// Open Modal
function openModal(imageSrc) {
    modal.style.display = 'block';
    modalImg.src = imageSrc;
    currentModalImage = imageSrc;
}

// Download Modal Image
function downloadModalImage() {
    if (!currentModalImage) return;

    const link = document.createElement('a');
    link.href = currentModalImage;
    link.download = `pokemon_card_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    showNotification('✅ Download started!');
}

// Batch Download via GET /generate/download
async function downloadBatch(format) {
    const numImages = parseInt(numImagesInput.value);
    const seed = seedInput.value ? `&seed=${parseInt(seedInput.value)}` : '';
    const formatParam = format === 'grid' ? '&format=grid' : '';
    const url = `${API_BASE_URL}/generate/download?num_images=${numImages}${seed}${formatParam}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Download failed');
        }

        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = objectUrl;
        link.download = format === 'grid' ? `pokemon_grid_${Date.now()}.png` : `pokemon_cards_${Date.now()}.zip`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(objectUrl);

        showNotification(`✅ ${format === 'grid' ? 'Grid PNG' : 'ZIP'} download started!`);
    } catch (error) {
        console.error('Download error:', error);
        showNotification(`❌ Error: ${error.message}`, true);
    }
}
// Show Notification
function showNotification(message, isError = false) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        background: ${isError ? 'var(--error-color)' : 'var(--success-color)'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        animation: slideIn 0.3s ease;
        font-weight: 600;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add animations to CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
