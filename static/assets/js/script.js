let uploadButtonFather = document.getElementByID("father-upload-button");
let chosenImageFather = document.getElementByID("father-chosen-image");
let fileNameFather = document.getElementByID("father-file-name");

let uploadButtonMother = document.getElementByID("mother-upload-button");
let chosenImageMother = document.getElementByID("mother-chosen-image");
let fileNameMother = document.getElementByID("mother-file-name");


uploadButtonFather.onchange = () => {
    let reader = new FileReader();
    reader.readAsDataURL(uploadButtonFather.files[0]);
    reader.onload = () => {
        chosenImageFather.setAttribute("src",reader.result);
    }
    fileNameFather.textContent = uploadButtonFather.files[0].name;
}

uploadButtonMother.onchange = () => {
    let reader = new FileReader();
    reader.readAsDataURL(uploadButtonMother.files[0]);
    reader.onload = () => {
        chosenImageMother.setAttribute("src",reader.result);
    }
    fileNameMother.textContent = uploadButtonMother.files[0].name;
}

window.onload = () => {
  for (let i of document.querySelectorAll(".gallery img")) {
    i.onclick = () => i.classList.toggle("full");
  }
};