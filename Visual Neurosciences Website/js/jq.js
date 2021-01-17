$(document).ready(function() {
  console.log("JQUERY");
  $(".submitBtn").click(function(e) {
    e.preventDefault();
    var name = $(".nameInp").val();
    var email = $(".emailInp").val();
    var message = $(".msg").val();
    var submit = $(".submitBtn").val();
    if (name == "" || email == "" || message == "") {
      $(".form-message").html("There is currently an error!");
      $("section").css("background", "rgba(0, 0, 0, 0.7)");
    } else {
      $("section").css("background", "rgba(0, 0, 0, 0.4)");
      $(".nameInp").val("");
      $(".emailInp").val("");
      $(".msg").val("");
      // $(".form-message").html("Your message has been sent!");
      $(".form-message").css("color", "#30ff34");
      $(".form-message").load("contact.php", {
        name: name,
        email: email,
        message: message,
        submit: submit
      });
    };
  });
});
