function navigateTo(section) {
    // Thay đổi trạng thái active cho nút menu
    const buttons = document.querySelectorAll(".menu-button");
    buttons.forEach(button => button.classList.remove("active"));

    const activeButton = document.querySelector(`.menu-button[onclick="navigateTo('${section}')"]`);
    if (activeButton) activeButton.classList.add("active");

    // Hiển thị thông báo tương ứng khi bấm nút
    alert(`Navigating to ${section} section!`);
}
