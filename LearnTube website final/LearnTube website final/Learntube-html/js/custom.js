// nav menu style
var nav = $("#navbarSupportedContent");
var btn = $(".custom_menu-btn");
btn.click
btn.click(function (e) {

    e.preventDefault();
    nav.toggleClass("lg_nav-toggle");
    document.querySelector(".custom_menu-btn").classList.toggle("menu_btn-style")
});


function getCurrentYear() {
    var d = new Date();
    var currentYear = d.getFullYear()

    $("#displayDate").html(currentYear);
}

getCurrentYear();



const faqItems = Array.from(document.querySelectorAll('.cs-faq-item'));
for (const item of faqItems) {
    const onClick = () => {
        item.classList.toggle('active');
    };
    item.addEventListener('click', onClick);
}



$(document).ready(function () {
  $('a.nav-link, a[href^="#"]').on('click', function (event) {
      if (this.hash !== '') {
          event.preventDefault();

          var target = $(this.hash);
          if (target.length) {
              $('html, body').animate(
                  {
                      scrollTop: target.offset().top
                  },
                  200, // Adjust speed here (400ms for fast, 200ms for ultra-fast)
                  'linear' // Ensures constant speed
              );
          }
      }
  });
});


  
  