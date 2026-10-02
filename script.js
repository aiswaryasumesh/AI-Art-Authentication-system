// ===============================
// AI ART AUTHENTICATOR
// Main JavaScript
// ===============================


// ---------- IMAGE UPLOAD ----------

const imageInput = document.getElementById("imageInput");
const cameraInput = document.getElementById("cameraInput");
const preview = document.getElementById("preview");
const previewImage = document.getElementById("previewImage");


// Choose Image
if (imageInput) {
    imageInput.addEventListener("change", function () {
        handleImage(this.files);
    });
}


// Open Camera
if (cameraInput) {
    cameraInput.addEventListener("change", function () {
        handleImage(this.files);
    });
}


// ---------- HANDLE IMAGE ----------

function handleImage(files) {

    if (!files || files.length === 0) {
        return;
    }

    const file = files[0];

    // Check whether the selected file is an image
    if (!file.type.startsWith("image/")) {
        alert("Please select a valid image.");
        return;
    }

    // Create image preview
    const imageURL = URL.createObjectURL(file);

    if (previewImage) {
        previewImage.src = imageURL;
    }

    if (preview) {
        preview.classList.remove("hidden");
    }
}


// ---------- START ANALYSIS ----------

function startAnalysis() {

    if (!previewImage || !previewImage.src) {
        alert("Please select an artwork first.");
        return;
    }

    location.href = "analysis.html";
}


// ---------- LOGIN ----------

const loginForm = document.getElementById("loginForm");

if (loginForm) {

    loginForm.addEventListener("submit", function (event) {

        event.preventDefault();

        const email = document.getElementById("email");
        const password = document.getElementById("password");

        if (!email || !password) {
            return;
        }

        if (email.value.trim() === "" || password.value.trim() === "") {
            alert("Please enter your email and password.");
            return;
        }

        // Demo login
        location.href = "home.html";
    });
}


// ---------- ANALYSIS PAGE ----------

const analysisPage = document.querySelector(".analysis-page");

if (analysisPage) {

    setTimeout(function () {

        location.href = "result.html";

    }, 5000);
}