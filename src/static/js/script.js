// Function to dynamically load sections and update the content
async function navigateTo(section) {
    // Update active button state
    const buttons = document.querySelectorAll(".menu-button");
    buttons.forEach((button) => button.classList.remove("active"));

    const activeButton = document.querySelector(`.menu-button[onclick="navigateTo('${section}')"]`);
    if (activeButton) activeButton.classList.add("active");

    // Fetch new content from the server
    const url = section === "" ? "/" : `/${section}`;
    try {
        const response = await fetch(url);
        if (response.ok) {
            const content = await response.text();

            // Parse and replace content in the container
            const tempDiv = document.createElement("div");
            tempDiv.innerHTML = content;
            const newContent = tempDiv.querySelector(".container");
            const mainContent = document.getElementById("content");

            if (newContent && mainContent) {
                mainContent.innerHTML = newContent.innerHTML;

                // Reinitialize events for dynamically loaded content
                initializeDynamicEvents();
            }
        } else {
            console.error(`Failed to load section: ${section}`);
        }
    } catch (error) {
        console.error("Error fetching section:", error);
    }
}

// Function to handle prediction form submission
async function submitPredictForm() {
    const form = document.getElementById("predictForm");
    const formData = new FormData(form);

    // Validate input values
    const age = formData.get("Age");
    const restingBP = formData.get("RestingBP");
    const cholesterol = formData.get("Cholesterol");
    const maxHR = formData.get("MaxHR");
    const oldpeak = formData.get("Oldpeak");

    if (age < 1 || age > 100) {
        alert("Age must be between 1 and 100.");
        return;
    }
    if (restingBP < 50 || restingBP > 200) {
        alert("Resting BP must be between 50 and 200.");
        return;
    }
    if (cholesterol < 100 || cholesterol > 600) {
        alert("Cholesterol must be between 100 and 600.");
        return;
    }
    if (maxHR < 60 || maxHR > 202) {
        alert("Max HR must be between 60 and 202.");
        return;
    }
    if (oldpeak < 0 || oldpeak > 6) {
        alert("Oldpeak must be between 0 and 6.");
        return;
    }

    // Submit valid data
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(Object.fromEntries(formData)),
    })
        .then((response) => response.json())
        .then((data) => {
            const resultElement = document.getElementById("result");
            const prediction = data.prediction;

            // Display result and change color based on the prediction
            resultElement.innerText = `Prediction: ${prediction}`;
            if (prediction === "Normal") {
                resultElement.style.color = "green"; // Green for normal
            } else if (prediction === "Heart Disease Detected") {
                resultElement.style.color = "red"; // Red for detected
            } else {
                resultElement.style.color = "#007bff"; // Default blue
            }
        })
        .catch((error) => console.error("Error:", error));
}

// Function to toggle parameter fields based on selected model
function toggleModelParams(selectedModel) {
    const xgboostParams = document.getElementById("xgboost-params");
    const neuralNetworkParams = document.getElementById("neural-network-params");

    if (selectedModel === "xgboost") {
        xgboostParams.style.display = "block";
        neuralNetworkParams.style.display = "none";
    } else if (selectedModel === "neural_network") {
        xgboostParams.style.display = "none";
        neuralNetworkParams.style.display = "block";
    }
}

async function trainModel() {
    console.log("trainModel function called");

    const modelType = document.getElementById("model-select").value;

    // Collect parameters based on the selected model
    const params =
        modelType === "xgboost"
            ? {
                  model: "xgboost",
                  n_estimators: document.getElementById("n_estimators").value,
                  learning_rate: document.getElementById("learning_rate").value,
                  max_depth: document.getElementById("max_depth").value,
                  subsample: document.getElementById("subsample").value,
              }
            : {
                  model: "neural_network",
                  epochs: document.getElementById("epochs").value,
                  learning_rate: document.getElementById("learning_rate_nn").value,
                  num_layers: document.getElementById("num_layers").value,
                  num_node: document.getElementById("num_node").value,
              };

    // Send training request to the backend
    try {
        const response = await fetch("/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Received data:", data);

        // Update images with the new results, adding timestamp to bypass cache
        document.getElementById("classification-report").src = data.classification_report + `?t=${Date.now()}`;
        document.getElementById("confusion-matrix").src = data.confusion_matrix + `?t=${Date.now()}`;
        document.getElementById("roc-auc").src = data.roc_auc + `?t=${Date.now()}`;

        alert("Training completed successfully!");
    } catch (error) {
        console.error("Error during training:", error);
        alert("Error during training. Please check the backend.");
    }
}

// Initialize dynamic events for loaded content
function initializeDynamicEvents() {
    const modelSelect = document.getElementById("model-select");
    if (modelSelect) {
        modelSelect.addEventListener("change", () => toggleModelParams(modelSelect.value));
        toggleModelParams(modelSelect.value); // Ensure the correct params are displayed on load
    }
}

// Ensure events are attached on initial load
document.addEventListener("DOMContentLoaded", initializeDynamicEvents);

// Hàm để hiển thị ảnh phóng to
// Hàm để hiển thị ảnh phóng to
function zoomImage(imageId) {
    const imgSrc = document.getElementById(imageId).src;

    // Tạo overlay nếu chưa có
    let overlay = document.querySelector(".image-overlay");
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.className = "image-overlay";
        overlay.innerHTML = `
            <img src="${imgSrc}" alt="Zoomed Image">
            <button class="close-button" onclick="closeZoom()">×</button>
        `;
        document.body.appendChild(overlay);
    } else {
        // Cập nhật ảnh trong overlay
        overlay.querySelector("img").src = imgSrc;
        overlay.style.display = "flex";
    }

    // Hiển thị overlay
    overlay.style.display = "flex";
}

// Hàm để đóng ảnh phóng to
function closeZoom() {
    const overlay = document.querySelector(".image-overlay");
    if (overlay) {
        overlay.style.display = "none";
    }
}
function closeModalWithAnimation() {
    const modal = document.getElementById("image-modal");
    const modalImage = document.getElementById("modal-image");
    const closeModal = document.querySelector(".close");

    // Ẩn modal và các thành phần bên trong
    modal.classList.remove("show");
    modalImage.classList.remove("show");
    closeModal.classList.remove("show");

    // Đợi animation kết thúc rồi mới ẩn modal
    setTimeout(() => {
        modal.style.display = "none";
    }, 300); // Thời gian này phải khớp với CSS transition
}

function initializeDynamicEvent() {
    const modal = document.getElementById("image-modal");
    const modalImage = document.getElementById("modal-image");
    const closeModal = document.querySelector(".close");
    if (modal && modalImage && closeModal) {
        // Gán sự kiện cho các phần tử nếu tồn tại
        document.querySelectorAll(".feature-image").forEach((img) => {
            img.addEventListener("click", function () {
                modal.style.display = "flex";
                modalImage.src = this.src;

                setTimeout(() => {
                    modal.classList.add("show");
                    modalImage.classList.add("show");
                    closeModal.classList.add("show");
                }, 10);
            });
        });

        closeModal.addEventListener("click", (e) => {
            e.stopPropagation();
            closeModalWithAnimation();
        });

        modal.addEventListener("click", (e) => {
            if (e.target === modal) {
                closeModalWithAnimation();
            }
        });

        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && modal.style.display === "flex") {
                closeModalWithAnimation();
            }
        });
    } 
}



// Gọi lại hàm sau khi nội dung thay đổi
document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM fully loaded and parsed");
    initializeDynamicEvent();
});
