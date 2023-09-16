document.addEventListener('DOMContentLoaded', function () {
    // Get the form element
    const predictionForm = document.getElementById('prediction-form');
  
    // Add an event listener to the form submit event
    predictionForm.addEventListener('submit', function (event) {
      // Prevent the default form submission behavior
      event.preventDefault();
  
      // Display the loading animation and hide the button text
      const predictButton = document.getElementById('predict-button');
      predictButton.querySelector('.button-text.default').style.display = 'none';
      predictButton.querySelector('.button-text.loading').style.display = 'inline';
  
      // Submit the form using JavaScript
      fetch('/predict', {
        method: 'POST',
        body: new FormData(predictionForm),
      })
        .then((response) => response.text())
        .then((data) => {
          // When the form submission is complete, hide the loading animation
          predictButton.querySelector('.button-text.default').style.display = 'inline';
          predictButton.querySelector('.button-text.loading').style.display = 'none';
  
          // Update the result
          const predictionResult = document.getElementById('prediction-result');
          predictionResult.textContent = data;
        })
        .catch((error) => {
          console.error('Error:', error);
        });
    });
  });
  





