$(document).ready(function() {
  //set up variables and constants
  var pageCounter = 1;
  const imagesArray = [];
  var currImagesIndex = 0;
  var currLabelIndex = 0;
  var currUser = "undefined";
  var excelName = "undefined";
  var imagesClassificationList = [["Grader", "Image", "Classification"]];
  var imagesLabels = [];
  //hide first pages two and three
  // $(".pageOne").hide();
  $(".pageTwo").hide();
  $(".pageThree").hide();

  //hide the warnings
  $(".error1").hide();
  $(".error2").hide();
  $(".error3").hide();
  $("#error4").hide();
  // $(".imageElement").attr("src", "https://static.wikia.nocookie.net/anchorman/images/e/ec/Brian-fantana.jpg/revision/latest/scale-to-width-down/340?cb=20120329161812")
  $("#errorPage2").hide();
  $("#checkmark1").hide();
  $("#checkmark2").hide();
  $("#checkmark3").hide();
  $(".fileUpload").hide();

  //click buttons page 1
  $(".usernameEnter").click(function() {
    registerName();
    $(".excelnameTag").focus();
  });
  $(".excelnameEnter").click(function() {
    registerExcel();
  });

  $(".fakeButton").click(function() {
    $(".fileUpload").click();
  });



  //keypresses page 1
  $(document).keypress(function(e) {
    const keypressed = e.keyCode;
    if (pageCounter == 1) {
      if (keypressed == 13) {
        if ($(".usernameTag").is(":focus")) {
          registerName();
        } else if ($(".excelnameTag").is(":focus")) {
          registerExcel();
        } else if ($("#checkmark1").is(":visible") && $("#checkmark2").is(":visible") && $("#checkmark3").is(":visible")) {
          $(".nextPage1").click();
        }
      } else {
        if ($(".usernameTag").is(":focus")) {

          $("#checkmark1").hide();
        } else if ($(".excelnameTag").is(":focus")) {
          $("#checkmark2").hide();
        }
      }
    }
  })



  //page 1 functions
  function registerName() {
    if ($(".usernameTag").val() != "") {
      if ($(".error1").is(":visible")) {
        $(".error1").hide();
      }
      $("#checkmark1").show();
      currUser = $(".usernameTag").val();
      $(".excelnameTag").focus();
    }
  }

  function registerExcel() {
    if ($(".excelnameTag").val() != "") {
      if ($(".error2").is(":visible")) {
        $(".error2").hide();
      }
      excelName = $(".excelnameTag").val() + ".csv";
      $("#checkmark2").show();
    }
  }





//file upload function
  var inps = document.querySelectorAll('.VHV');
  [].forEach.call(inps, function(inp) {
    inp.onchange = function(e) {
      if ($(".errorThree").parentNode != null) {
        $(".errorThree").hide();
      }
    for (var i = 0; i < this.files.length; i++) {
      imagesArray.push(this.files[i].webkitRelativePath);
    };
    // console.log(imagesArray);
    $("#checkmark3").show();
    $(".error3").hide();
    };
  });




  //page 1 to 2 function
  $(".nextPage1").click(function() {
    var pageOneErrors = checkPageOneInputs();
    // var pageOneErrors = 0;
    // registerName();
    // registerExcel();
    if (pageOneErrors == 0) {
      $(".pageOne").hide();
      $(".pageTwo").show();
      pageCounter += 1;
    }
  })
  $(".fileUpload").change(function() {
    for (var i = 0; i < this.files.length; i++) {
      imagesArray.push(this.files[i]);
    }
    $("#checkmark3").show();
  })
  function checkPageOneInputs() {
    var errorCount = 0;
    if (!$("#checkmark1").is(":visible")) {
      $(".error1").show();
      errorCount += 1;
    }
    if (!$("#checkmark2").is(":visible")) {
      $(".error2").show();
      errorCount += 1;
    }
    if (!$("#checkmark3").is(":visible")) {
      $(".error3").show();
      errorCount += 1;
    }
    return errorCount;
  }




  //page2 buttons

  $(".enterButton2").click(function() {
    if ($(".input2").val() != "") {
      appendListItem();
    }
    $(".input2").focus();
  });

  $(".nextPage2").click(function() {
    if (checkPageTwoLabels()) {
      page2to3();
    }
  });
  $(".prevButtonPage2To1").mousedown(function() {
    page2to1();
  })


  //page2 functions
  function appendListItem() {
    var divToAdd = $("<div class='addedDiv'></div>");
    var textDiv = $("<div class='textClass'></div>").text($(".input2").val());
    var deleteDiv = $("<div class='deleteDiv'></div>").text("Delete");
    divToAdd.append(textDiv, deleteDiv);
    $(".buttonsPart").append(divToAdd);
    imagesLabels.push($(".input2").val());
    $(".input2").val("");
    $("#errorPage2").hide();
    // console.log(imagesLabels);
    $(".input2").focus();
  }

  function checkPageTwoLabels() {
    if (imagesLabels.length == 0) {
      $("#errorPage2").show();
      return false;
    }
    return true;
  }

  function loadButtonLabels() {
    for (var i = 0; i < imagesLabels.length; i++) {
      // console.log(i);
      const index = i;
      const newButton = $("<button></button>").text(i.toString() + ": " + imagesLabels[index]);
      $(".buttons").append(newButton);
      newButton.mousedown(function() {
        const tempVal = $(".elInput").val();
        if (tempVal != "") {
          const newVal = tempVal + ", " + imagesLabels[index];
          $(".elInput").val(newVal);
        } else {
          $(".elInput").val(imagesLabels[index]);
        }
      })
    }
  }
  function loadLabelsPage2() {
    for (var i = 0; i < imagesLabels.length; i++) {
      const newLabel = $("<div></div>").text(i.toString() + ": " + imagesLabels[i]);
      $(".buttonsPart").append(newLabel);
    }
  }
  function page2to3() {
    $(".pageTwo").hide();
    $(".pageThree").show();
    displayImage(imagesArray[0]);
    loadButtonLabels();
    pageCounter += 1;
  }
  function page2to1() {
    $(".pageOne").show();
    $(".pageTwo").hide();
    pageCounter -= 1;
  }
  //page2 keypresses
  $(document).keypress(function(e) {
    const keypressed = e.keyCode;
    // pageCounter = 2;
    if (keypressed == 13) {
      if (pageCounter == 2) {
        // console.log("Got here");
        if ($(".input2").is(":focus")) {
          if ($(".input2").val() != "") {
            appendListItem();
          } else {
            if (imagesLabels.length != 0) {
              // console.log(imagesLabels.length);
              page2to3();
            }
          }
        }
      }
    }
  });

  //page3

  //page3 buttons
  $(".previousButton3").click(function() {
    currImagesIndex -= 1;
    if (currImagesIndex < 0) {
      currImagesIndex = imagesArray.length - 1
    }
    displayImage(imagesArray[currImagesIndex]);
    // $(".imageElement").attr("src", imagesArray[currImagesIndex]);
  });

  $(".nextButton3").click(function() {
    nextImage();
  });

  $(".enterButton3").click(function() {
    enterFuncPage3();
  })
  $(".prevButtonPage3To2").click(function() {
    page3to2();
  });
  $(".tabulate").click(function() {
    exportToCsv(excelName, imagesClassificationList);
  })
  $(".clearButton3").click(function() {
    $(".elInput").val("");
  })


  //functions page 3
  function page3to2() {
    // console.log("page3to2");
    // loadLabelsPage2();
    $(".pageThree").hide();
    $(".pageTwo").show();
    $(".buttons").children().remove();
    pageCounter -= 1;
  }
  function nextImage() {
    currImagesIndex += 1;
    if (currImagesIndex == imagesArray.length) {

      currImagesIndex = 0;
    }
    displayImage(imagesArray[currImagesIndex]);
    for (var i = 0; i < imagesClassificationList.length; i++) {
      console.log(imagesClassificationList[i][1]);
      console.log(imagesArray[currImagesIndex].name);
      if (imagesClassificationList[i][1] == imagesArray[currImagesIndex].name) {
        $(".elInput").val(imagesClassificationList[i][2])
      }
    }

    // $(".imageElement").attr("src", imagesArray[currImagesIndex]);
  }

  function displayImage(imageFile) {
    const reader = new FileReader();
    reader.addEventListener("load", function() {
      $(".imageElement").attr("src", this.result);
    });
    reader.readAsDataURL(imageFile);
  }
  function enterFuncPage3() {
    if ($(".elInput").val() == "") {
      $("#error4").show();
    } else {
      $("#error4").hide();
      const temp = [];
      temp.push(currUser);
      const y = imagesArray[currImagesIndex].name;
      console.log(y);
      temp.push(y);
      temp.push($(".elInput").val());
      // console.log(temp);
      imagesClassificationList.push(temp);
      // console.log(imagesClassificationList);
      $(".elInput").val("");
      $(".elInput").focus();
      nextImage();
    }

  }

  //download function
  function exportToCsv(filename, rows) {
    // console.log("filename is " + filename)
    var processRow = function (row) {
        var finalVal = '';
        for (var j = 0; j < row.length; j++) {
            var innerValue = row[j] === null ? '' : row[j].toString();
            if (row[j] instanceof Date) {
                innerValue = row[j].toLocaleString();
            };
            var result = innerValue.replace(/"/g, '""');
            if (result.search(/("|,|\n)/g) >= 0)
                result = '"' + result + '"';
            if (j > 0)
                finalVal += ',';
            finalVal += result;
        }
        return finalVal + '\n';
    };

    var csvFile = '';
    for (var i = 0; i < rows.length; i++) {
        csvFile += processRow(rows[i]);
    }

    var blob = new Blob([csvFile], { type: 'text/csv;charset=utf-8;' });
    if (navigator.msSaveBlob) { // IE 10+
        navigator.msSaveBlob(blob, filename);
    } else {
        var link = document.createElement("a");
        if (link.download !== undefined) { // feature detection
            // Browsers that support HTML5 download attribute
            var url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
  }
  //keyEvents page 3
  $(document).keypress(function(e) {
    const keypressed = e.keyCode;
    if (pageCounter == 3) {
      if (keypressed == 13) {
        if ($(".elInput").val() != "") {
          enterFuncPage3();
          document.activeElement.blur();
        }
      } else if (keypressed >= 48 && keypressed <= 57) {
        if (!$(".elInput").is(":focus")) {
          if (keypressed-48 < imagesLabels.length) {
            const index = keypressed - 48;
            const tempVal = $(".elInput").val();
            if (tempVal != "") {
              const newVal = tempVal + ", " + imagesLabels[index];
              $(".elInput").val(newVal);
            } else {
              $(".elInput").val(imagesLabels[index]);
            }
          }
        }
      }
    }
  });


  //overall page functions
  const allButtons = document.querySelectorAll("button");
  allButtons.forEach(button => {
    button.addEventListener('click', () => {
      document.activeElement.blur();
      button.style.color = "rgba(250, 165, 45, 0.9)";
      setTimeout(function(){
        button.style.transition = ".5s";
        button.style.color = "rgba(254, 251, 239, 0.6)";  // Change the color back to the original
      }, 400);
    });

  })
});
