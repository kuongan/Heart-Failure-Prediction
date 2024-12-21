async function navigateTo(section) {
    // Cập nhật trạng thái active của các nút
    const buttons = document.querySelectorAll(".menu-button");
    buttons.forEach(button => button.classList.remove("active"));

    const activeButton = document.querySelector(`.menu-button[onclick="navigateTo('${section}')"]`);
    if (activeButton) activeButton.classList.add("active");

    // Fetch nội dung từ server
    let url = section === "" ? "/" : `/${section}`;
    const response = await fetch(url);

    if (response.ok) {
        const content = await response.text();
        const tempDiv = document.createElement("div");
        tempDiv.innerHTML = content;

        // Trích lấy nội dung chính từ response
        const newContent = tempDiv.querySelector(".container");
        document.getElementById("content").innerHTML = newContent.innerHTML;
    } else {
        document.getElementById("content").innerHTML = "<h2>Error loading content</h2>";
    }
}

async function submitPredictForm() {
    const form = document.getElementById("predictForm");
    const formData = new FormData(form);

    // Kiểm tra các giá trị
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

    // Submit nếu dữ liệu hợp lệ
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(Object.fromEntries(formData)),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = `Prediction: ${data.prediction}`;
    })
    .catch(error => console.error("Error:", error));
}
