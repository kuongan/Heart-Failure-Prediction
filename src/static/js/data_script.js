const modal = document.getElementById("image-modal");
const modalImage = document.getElementById("modal-image");
const closeModal = document.querySelector(".close");

// Thêm sự kiện click cho tất cả ảnh
document.querySelectorAll(".feature-image").forEach((img) => {
    img.addEventListener("click", function() {
        modal.style.display = "flex";
        modalImage.src = this.src;
        
        // Thêm timeout nhỏ để đảm bảo transition hoạt động
        setTimeout(() => {
            modal.classList.add("show");
            modalImage.classList.add("show");
            closeModal.classList.add("show");
        }, 10);
    });
});

// Hàm đóng modal
function closeModalWithAnimation() {
    modal.classList.remove("show");
    modalImage.classList.remove("show");
    closeModal.classList.remove("show");
    
    // Đợi animation kết thúc rồi mới ẩn modal
    setTimeout(() => {
        modal.style.display = "none";
    }, 300); // 300ms = thời gian transition
}

// Đóng modal khi click nút close
closeModal.addEventListener("click", (e) => {
    e.stopPropagation();
    closeModalWithAnimation();
});

// Đóng modal khi click bên ngoài ảnh
modal.addEventListener("click", (e) => {
    if (e.target === modal) {
        closeModalWithAnimation();
    }
});

// Thêm phím tắt ESC để đóng modal
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.style.display === "flex") {
        closeModalWithAnimation();
    }
});