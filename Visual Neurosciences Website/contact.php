<?php
session_start();
if (isset($_POST["submit"])) {
  $name = $_POST["name"];
  $email = $_POST["email"];
  $message = $_POST["message"];
  $_SESSION["user"] = $name;
  $error = 0;
  $_POST["error"] = 0;
  echo "Your message has been sent!";
  // echo $message;
  // header("location: ./index.php?error=none");
  // mail("milealeonard@gmail.com","Message from website contact ".$email." from ".$name, $message);
} else {
  echo "step2";
}
