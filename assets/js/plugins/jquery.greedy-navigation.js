/*
* Greedy Navigation
*
* http://codepen.io/lukejacksonn/pen/PwmwWV
*
*/

var $nav = $('#site-nav');
var $btn = $('#site-nav button');
var $vlinks = $('#site-nav .visible-links');
var $hlinks = $('#site-nav .hidden-links');

function updateNav() {
  if (!$nav.length) return;
  // Re-measure from the complete list so resizing can restore every link.
  $hlinks.children().appendTo($vlinks);
  $btn.addClass('hidden').removeClass('close').attr('aria-expanded', 'false');
  $hlinks.addClass('hidden');
  if ($vlinks.outerWidth() > $nav.width()) {
    $btn.removeClass('hidden');
    var availableSpace = $nav.width() - $btn.outerWidth() - 16;
    var movable = $vlinks.children('*:not(.masthead__menu-item--lg)');
    while ($vlinks.outerWidth() > availableSpace && movable.length) {
      movable.last().prependTo($hlinks);
      movable = $vlinks.children('*:not(.masthead__menu-item--lg)');
    }
  }
  $btn.attr('count', $hlinks.children().length);
}

// Window listeners

$(window).resize(function() {
  updateNav();
});

$btn.on('click', function() {
  $hlinks.toggleClass('hidden');
  $(this).toggleClass('close');
  $(this).attr('aria-expanded', String(!$hlinks.hasClass('hidden')));
});

$(document).on('keydown', function(event) {
  if (event.key === 'Escape' && !$hlinks.hasClass('hidden')) {
    $hlinks.addClass('hidden');
    $btn.removeClass('close').attr('aria-expanded', 'false').trigger('focus');
  }
});

$(window).on('load', updateNav);
updateNav();
