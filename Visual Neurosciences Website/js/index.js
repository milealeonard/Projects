window.addEventListener("load", () => {
  const dan = document.querySelectorAll(".dan > img")
  const imgtext = document.querySelectorAll(".imgtext");
  var framesArray = [];
  var danArray = [];
  var imgArray = [];
  var counter = 0;
  dan.forEach(spinDan);

  function spinDan(dan) {
    danArray.push(dan);
  }

  imgtext.forEach(imgtextFunc);
  function imgtextFunc(imgtxt) {
    imgArray.push(imgtxt);
  }
  function mouseOverFunc(danInstance, imgTxt) {
    danInstance.style.transition = "1s";
    danInstance.style.filter = "brightness(30%)";
    danInstance.style.transform = "scale(2)";
    imgTxt.style.transition = "0.5s";
    imgTxt.style.opacity = "100%";
    for (var i = 0; i < danArray.length; i++) {
      const danInstanceTwo = danArray[i];
      if (danInstanceTwo != danInstance) {
        danInstanceTwo.style.transition = "1s";
        danInstanceTwo.style.opacity = "0%";
      }
    }
  }
  for (var i = 0; i < dan.length; i++) {
    const danInstance = danArray[i];
    const imgTxt = imgArray[i];
    danInstance.addEventListener("mouseover", function() {
      mouseOverFunc(danInstance, imgTxt);
    });
    danInstance.addEventListener("mouseleave", function() {
      // console.log(frame);
      // danInstance.style.transition = "1s";
      danInstance.style.filter = "brightness(100%)";
      // console.log(fifi);
      danInstance.style.transform = "";
      console.log(danInstance.style.transform);
      // danInstance.style.transform += "rotate(360deg)";
      imgTxt.style.opacity = "0%";
      for (var i = 0; i < danArray.length; i++) {
        const danInstanceTwo = danArray[i];
        if (danInstanceTwo != danInstance) {
          // danInstanceTwo.style.transition = "1s";
          danInstanceTwo.style.opacity = "100%";
        }
      }
      // dan.style.transform -= "rotate(180deg)";
      // frame.style.transform += "rotate(-360deg)";
    });
    imgTxt.addEventListener("mouseover", function() {
      mouseOverFunc(danInstance, imgTxt);
    });
  }
  // const classifEyeHover = document.querySelector(".classifeyeLink");
  // classifEyeHover.addEventListener("mouseover", () => {
  //   classifEyeHover.style.color = "red";
  // });
  // classifEyeHover.addEventListener("mouseleave", () => {
  //   classifEyeHover.style.color = "white";
  // });
  const formElements = document.querySelectorAll("#form");
  const submitBtn = document.querySelector(".submitBtn");
  const nameInp = document.querySelector(".nameInp");
  const emailInp = document.querySelector(".emailInp");
  const messageInp = document.querySelector(".messageInp");
  submitBtn.addEventListener("onclick", () => {
    console.log(nameInp);
  })
})
