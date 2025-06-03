// Function to handle Employees link click
document.getElementById("employees-link").addEventListener("click", function(event) {
    event.preventDefault();
    loadSidebar(); // Load the sidebar
});

// Function to load the sidebar (from sidebar.html)
function loadSidebar() {
    fetch('sidebar.html')
        .then(response => response.text())
        .then(data => {
            document.getElementById("sidebar-container").innerHTML = data;
            document.getElementById("sidebar").classList.add("show");
        })
        .catch(error => {
            console.log('Error loading sidebar:', error);
        });
}

// Dropdown Toggle Functionality
function toggleDropdown() {
    document.getElementById("dropdown-content").classList.toggle("show");
}

// Close dropdown when clicking outside
window.onclick = function(event) {
    if (!event.target.matches('.dropdown-btn')) {
        var dropdowns = document.getElementsByClassName("dropdown-content");
        for (var i = 0; i < dropdowns.length; i++) {
            dropdowns[i].classList.remove('show');
        }
    }
}
