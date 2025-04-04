<?php
// index.php: Load ontouml models CSV, embeddings CSVs, and render the interface

// --- Load ontouml_models.csv ---
$csvData = array();
if (($handle = fopen("../processing/ontouml_models.csv", "r")) !== FALSE) {
    $header = fgetcsv($handle);
    while (($row = fgetcsv($handle)) !== FALSE) {
        $csvData[] = array_combine($header, $row);
    }
    fclose($handle);
}
// For each row, load images from both new-diagrams and old-diagrams folders
foreach ($csvData as &$row) {
    $key = $row['key'];
    $dirNew = "../datasets/ontouml-models-master/models/{$key}/new-diagrams/";
    $dirOld = "../datasets/ontouml-models-master/models/{$key}/old-diagrams/";
    $imagesNew = is_dir($dirNew) ? glob($dirNew . "*.{jpg,jpeg,png}", GLOB_BRACE) : array();
    $imagesOld = is_dir($dirOld) ? glob($dirOld . "*.{jpg,jpeg,png}", GLOB_BRACE) : array();
    $row['images'] = array_merge($imagesNew, $imagesOld);
}
unset($row);

// Group models under a fixed key "ontoUML" by theme (sorted alphabetically)
$groupedData = array('ontoUML' => array());
foreach ($csvData as $row) {
    $theme = $row['theme'];
    if (!isset($groupedData['ontoUML'][$theme])) {
        $groupedData['ontoUML'][$theme] = array();
    }
    $groupedData['ontoUML'][$theme][] = array(
        'title' => $row['title'],
        'key'   => $row['key']
    );
}
ksort($groupedData['ontoUML']);

// Build a mapping for CSV pages ? each page id is "page-" plus the model key.
$csvPagesData = array();
foreach ($csvData as $row) {
    $pageId = 'page-' . $row['key'];
    $csvPagesData[$pageId] = $row;
}

// --- Load Embeddings Data ---
// Alignment results (for both aligned NLT and aligned LLM)
$embeddingsData = array();
if (($handle = fopen("../processing/ontouml_embeddings_2d.csv", "r")) !== FALSE) {
    $headerEmb = fgetcsv($handle);
    while (($row = fgetcsv($handle)) !== FALSE) {
         $embeddingsData[] = array_combine($headerEmb, $row);
    }
    fclose($handle);
}
// LLM embeddings (used for Embeddings Comparison)
$llmEmbeddingsData = array();
if (($handle = fopen("../processing/ontouml_llm_embeddings.csv", "r")) !== FALSE) {
    $headerLlm = fgetcsv($handle);
    while (($row = fgetcsv($handle)) !== FALSE) {
         $llmEmbeddingsData[] = array_combine($headerLlm, $row);
    }
    fclose($handle);
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Aligning AI Model's Knowledge and Conceptual Model's Symbolism</title>
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
  <!-- Bootstrap Icons -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
  <!-- Plotly Library for Interactive Visualization -->
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <!-- jQuery -->
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <style>
    /* Global styles */
    html, body {
      font-size: 12px;
      line-height: 1.2;
      margin: 0;
      padding: 0;
      overflow: hidden;
    }
    *, *::before, *::after { box-sizing: border-box; }
    .content-wrapper {
      height: calc(100vh - 75px);
      display: flex;
      flex-direction: column;
    }
    .tab-area {
      border: 1px solid #dee2e6;
      overflow: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    .tab-area .tab-content { padding: 5px; }
    .nav-tabs .nav-link {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 1em;
      padding: 3px 6px;
    }
    .tab-area .nav-tabs {
      position: sticky;
      top: 0;
      z-index: 1;
      background-color: #fff;
      border-bottom: 1px solid #dee2e6;
    }
    /* Left Sidebar */
    #leftPanel {
      border-right: 1px solid #dee2e6;
      height: 100%;
      overflow: auto;
      padding: 5px;
    }
    .accordion-item { margin: 0; }
    .accordion-header { font-size: 1em; padding: 3px 6px; }
    .accordion-button {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      font-size: 1em;
      padding: 3px 6px;
    }
    .accordion-body { font-size: 1em; padding: 5px; }
    .accordion-body a { text-decoration: none; }
    .default-content { font-style: italic; color: #777; }
    .navbar-brand img { height: 20px; }
    /* Chat Panel */
    #chatPanel { border-top: 1px solid #dee2e6; padding-top: 5px; }
    #chatControls input, #chatControls textarea { width: 100%; font-size: 1em; }
    #toastContainer { position: fixed; bottom: 0; right: 0; z-index: 1055; }
    .inline-control { display: inline-block; vertical-align: middle; margin-right: 5px; font-size: 1em; }
    .thumbnail-image { cursor: pointer; }
    /* Model Detail Page Styling */
    h1 { font-size: 24px; margin-bottom: 10px; }
    hr { border-top: 1px solid #ccc; margin: 10px 0; }
    /* Diagram images */
    .diagram-image { width: 100%; max-height: 400px; object-fit: contain; }
    /* Plotly container */
    #plotlyDiv { width: 100%; height: 600px; }
    /* Embeddings tooltip */
    #embeddingsTooltip {
      position: absolute;
      background: rgba(255,255,255,0.9);
      border: 1px solid #ccc;
      padding: 5px;
      font-size: 12px;
      pointer-events: none;
      display: none;
      z-index: 2000;
    }
  </style>
</head>
<body class="d-flex flex-column">
  <!-- Top Navigation Bar (omitting repetitive menu items for brevity) -->
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container-fluid">
      <a class="navbar-brand d-flex align-items-center" href="#">
        <img src="logo.png" alt="Logo" class="me-2" />
        Aligning AI Model's Knowledge and Conceptual Model's Symbolism
      </a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarContent">
        <span class="navbar-toggler-icon"></span>
      </button>
      <!-- (Menu items omitted for brevity) -->
      <div class="d-flex">
        <button class="btn btn-secondary" type="button" data-bs-toggle="modal" data-bs-target="#modalHelp" data-key="H" title="Help (Alt+H)">
          <i class="bi bi-question-circle"></i>
        </button>
      </div>
    </div>
  </nav>

  <!-- Main Content Area -->
  <div class="container-fluid content-wrapper flex-grow-1">
    <div class="row h-100">
      <!-- Left Sidebar: Fixed group "OntoUML" with sorted accordions by theme -->
      <div class="col-2" id="leftPanel">
        <h5 class="mt-2">OntoUML</h5>
        <div class="accordion" id="ontoumlAccordion">
          <?php foreach ($groupedData['ontoUML'] as $theme => $items): ?>
            <div class="accordion-item">
              <h2 class="accordion-header" id="heading-<?php echo preg_replace("/\s+/", "", $theme); ?>">
                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapse-<?php echo preg_replace("/\s+/", "", $theme); ?>">
                  <?php echo htmlspecialchars($theme); ?>
                </button>
              </h2>
              <div id="collapse-<?php echo preg_replace("/\s+/", "", $theme); ?>" class="accordion-collapse collapse" data-bs-parent="#ontoumlAccordion">
                <div class="accordion-body">
                  <ul class="list-unstyled">
                    <?php foreach ($items as $item): ?>
                      <li>
                        <a href="#" class="menu-item-left"
                           data-title="<?php echo htmlspecialchars($item['title']); ?>"
                           data-page="<?php echo 'page-' . htmlspecialchars($item['key']); ?>"
                           data-bs-toggle="tooltip" title="Click to view details">
                          <?php echo htmlspecialchars($item['title']); ?>
                        </a>
                      </li>
                    <?php endforeach; ?>
                  </ul>
                </div>
              </div>
            </div>
          <?php endforeach; ?>
        </div>
      </div>

      <!-- Center Column with Tab Areas -->
      <div class="col p-0 d-flex flex-column">
        <!-- Upper Tab Area (main) -->
        <div id="upperTabArea" class="tab-area d-flex flex-column">
          <ul class="nav nav-tabs" id="mainTab" role="tablist"></ul>
          <div class="tab-content flex-grow-1" id="mainTabContent">
            <div class="default-content text-center p-3">
              No tabs are open. Please select an option from the sidebar to get started.
            </div>
          </div>
        </div>
        <!-- Lower Tab Area (extra) -->
        <div id="lowerTabArea" class="tab-area d-flex flex-column">
          <ul class="nav nav-tabs" id="extraTabHolder" role="tablist"></ul>
          <div class="tab-content flex-grow-1" id="extraTabContent">
            <div class="default-content text-center p-3">
              No tabs are open. Please select an option from the sidebar to get started.
            </div>
          </div>
        </div>
      </div>

      <!-- Right Sidebar: Accordions for Embeddings and Alignment Results -->
      <div class="col-3 p-0" id="rightPanel">
        <div id="rightSidebar">
          <div id="nestedMenu">
            <h5>AI Models</h5>
            <div class="accordion" id="rightAccordion">
              <!-- Embeddings Comparison Accordion -->
              <div class="accordion-item">
                <h2 class="accordion-header" id="headingComparison">
                  <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseComparison">
                    Embeddings Comparison
                  </button>
                </h2>
                <div id="collapseComparison" class="accordion-collapse collapse show" data-bs-parent="#rightAccordion">
                  <div class="accordion-body">
                    <ul class="list-unstyled">
                      <li>
                        <a href="#" class="menu-item-right"
                           data-title="NLT to CMT"
                           data-page="embComp_nlt"
                           data-bs-toggle="tooltip" title="View NLT to CMT embeddings comparison">
                          NLT to CMT
                        </a>
                      </li>
                      <li>
                        <a href="#" class="menu-item-right"
                           data-title="LLM to CMT"
                           data-page="embComp_llm"
                           data-bs-toggle="tooltip" title="View LLM to CMT embeddings comparison">
                          LLM to CMT
                        </a>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
              <!-- Alignment Results Accordion -->
              <div class="accordion-item">
                <h2 class="accordion-header" id="headingAlignment">
                  <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseAlignment">
                    Alignment Results
                  </button>
                </h2>
                <div id="collapseAlignment" class="accordion-collapse collapse" data-bs-parent="#rightAccordion">
                  <div class="accordion-body">
                    <ul class="list-unstyled">
                      <li>
                        <a href="#" class="menu-item-right"
                           data-title="Aligned NLT to CMT"
                           data-page="align_nlt"
                           data-bs-toggle="tooltip" title="View aligned NLT to CMT results">
                          Aligned NLT to CMT
                        </a>
                      </li>
                      <li>
                        <a href="#" class="menu-item-right"
                           data-title="Aligned LLM to CMT"
                           data-page="align_llm"
                           data-bs-toggle="tooltip" title="View aligned LLM to CMT results">
                          Aligned LLM to CMT
                        </a>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <!-- Chat Panel -->
          <div id="chatPanel">
            <h5>Help Assistant</h5>
            <div class="mb-2">
              <label for="openaiKey" class="form-label">AI?powered Chat Assistant (Not available during the double-blinded review phase)</label>
              <input type="text" id="openaiKey" class="form-control" placeholder="Enter your OpenAI key">
            </div>
            <div id="chatContent">
              <p>Select a tab to chat about it.</p>
            </div>
            <div id="chatControls">
              <div class="mb-2">
                <textarea id="chatInput" class="form-control" placeholder="Ask a question about the selected tab... (Not available during the double-blinded review phase)"></textarea>
              </div>
              <button id="sendChat" class="btn btn-primary">Send</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Modal for Enlarged Image -->
  <div class="modal fade" id="imageModal" tabindex="-1" aria-labelledby="imageModalLabel" aria-hidden="true">
    <div class="modal-dialog modal-dialog-centered modal-lg">
      <div class="modal-content">
        <div class="modal-body">
          <img id="modalImage" src="" class="img-fluid" alt="Enlarged Diagram" />
        </div>
      </div>
    </div>
  </div>

  <!-- Tooltip for Embeddings Visualization -->
  <div id="embeddingsTooltip"></div>

  <!-- Toast Container -->
  <div id="toastContainer"></div>

  <!-- Footer with Collapsible Switch (omitted for brevity) -->
  <footer class="bg-secondary text-white p-2">
    <div class="container-fluid">
      <!-- Footer content here (unchanged) -->
      <div class="row align-items-center">
        <div class="col-auto">
          <button class="btn btn-sm btn-secondary" type="button" data-bs-toggle="modal" data-bs-target="#modalLeftAction" data-key="L" title="Left Action (Alt+L)">
            <i class="bi bi-star"></i>
          </button>
        </div>
        <div class="col text-center">
          <div class="form-check form-switch d-inline-block">
            <input class="form-check-input" type="checkbox" id="toggleExtra" checked>
            <label class="form-check-label" for="toggleExtra">Split View</label>
          </div>
          <div class="form-check form-switch d-inline-block">
            <input class="form-check-input" type="checkbox" id="toggleAssistant" checked>
            <label class="form-check-label" for="toggleAssistant"> AI Assistant</label>
          </div>
          <div class="form-check form-switch d-inline-block">
            <input class="form-check-input" type="checkbox" id="toggleLeftPanel" checked>
            <label class="form-check-label" for="toggleLeftPanel"> Left Sidebar</label>
          </div>
          <div class="form-check form-switch d-inline-block">
            <input class="form-check-input" type="checkbox" id="toggleRightPanel" checked>
            <label class="form-check-label" for="toggleRightPanel"> Right Sidebar</label>
          </div>
        </div>
        <div class="col-auto d-flex align-items-center">
          <div class="me-2">Status: Ready</div>
          <button class="btn btn-sm btn-secondary" type="button" data-bs-toggle="modal" data-bs-target="#modalRightAction" data-key="R" title="Right Action (Alt+R)">
            <i class="bi bi-info-circle"></i>
          </button>
        </div>
      </div>
    </div>
  </footer>

  <!-- (Modals for Settings, Help, Left Action, Right Action omitted for brevity) -->

  <!-- Bootstrap Bundle JS -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <!-- Embedded Data and Interactive Visualization Scripts -->
  <script>
    // Utility function to clean values by removing surrounding brackets and quotes
    function cleanValue(value) {
      value = value.trim();
      if (value.startsWith("[") && value.endsWith("]")) {
         value = value.substring(1, value.length - 1);
      }
      return value.replace(/['"]+/g, "");
    }

    // Global data loaded from PHP
    var csvPagesData = <?php echo json_encode($csvPagesData); ?>;
    var embeddingsData = <?php echo json_encode($embeddingsData); ?>;
    var llmEmbeddingsData = <?php echo json_encode($llmEmbeddingsData); ?>;
    let tabCounter = 0;

    function updateDefaultContentForArea(target) {
      if (target === 'main') {
        if ($("#mainTab li").length === 0) {
          $("#mainTabContent").html('<div class="default-content text-center p-3">No tabs are open. Please select an option from the sidebar to get started.</div>');
        }
      } else {
        if ($("#extraTabHolder li").length === 0) {
          $("#extraTabContent").html('<div class="default-content text-center p-3">No tabs are open. Please select an option from the sidebar to get started.</div>');
        }
      }
    }

    function initializeTooltips() {
      $('[data-bs-toggle="tooltip"]').each(function() {
        if (!bootstrap.Tooltip.getInstance(this)) {
          new bootstrap.Tooltip(this, { trigger: 'hover' });
        }
      });
    }
    initializeTooltips();

    // Render page content ? for model details pages and for embeddings visualizations
    function renderPageContent(pageId) {
      // For embedding visualizations, return a div for Plotly
      if (pageId === "embComp_nlt" || pageId === "embComp_llm" || pageId === "align_nlt" || pageId === "align_llm") {
         return "<div id='plotlyDiv'></div>";
      } else if (csvPagesData && csvPagesData[pageId]) {
         var row = csvPagesData[pageId];
         var html = "<div class='row'>";
         // Left column: Render model details
         html += "<div class='col-md-6'>";
         var labelMapping = {
           key: "Key",
           title: "Title",
           theme: "Theme",
           ontologyType: "Ontology Type",
           designedForTask: "Designed For Task",
           language: "Language",
           context: "Context",
           source: "Source",
           keywords: "Keywords"
         };
         for (var field in row) {
           if (field === "images") continue;
           var value = cleanValue(row[field]);
           var label = labelMapping[field] || (field.charAt(0).toUpperCase() + field.slice(1));
           if (field === "title") {
              html += "<h1>" + value + "</h1><hr/>";
           } else if (field === "keywords" || field === "source") {
              var items = value.split(",").map(item => cleanValue(item)).filter(item => item.length > 0);
              if (items.length > 0) {
                  html += "<p><strong>" + label + ":</strong></p><ul>";
                  items.forEach(function(item) {
                      if (field === "source") {
                         html += "<li><a href='" + item + "' target='_blank'>" + item + "</a></li>";
                      } else {
                         html += "<li>" + item + "</li>";
                      }
                  });
                  html += "</ul><hr/>";
              }
           } else {
              html += "<p><strong>" + label + ":</strong> " + value + "</p><hr/>";
           }
         }
         html += "</div>";
         // Right column: Render diagrams
         html += "<div class='col-md-6'>";
         if (row.images && row.images.length > 0) {
              html += "<h4>Diagrams</h4><hr/><div class='row'>";
              row.images.forEach(function(imgSrc) {
                  html += "<div class='col-12 mb-2'><img src='" + imgSrc + "' class='img-fluid img-thumbnail diagram-image thumbnail-image' /></div>";
              });
              html += "</div>";
         } else {
              html += "<p>No diagrams available.</p>";
         }
         html += "</div></div>";
         return html;
      } else {
         return "<p>No data available for this page.</p>";
      }
    }

    // Modified addNewTab to accept an extra parameter pageId
    function addNewTab(title, content, target, pageId) {
      tabCounter++;
      const tabId = 'tab' + tabCounter;
      let navSelector, contentSelector;
      if (target === 'main') {
        navSelector = "#mainTab";
        contentSelector = "#mainTabContent";
      } else {
        navSelector = "#extraTabHolder";
        contentSelector = "#extraTabContent";
      }
      $(contentSelector).find('.default-content').remove();
      const tabTemplate = `
        <li class="nav-item" role="presentation">
          <button class="nav-link" id="${tabId}-tab" data-bs-toggle="tab" data-bs-target="#${tabId}" type="button" role="tab">
            ${title} <span class="close-tab" style="margin-left:5px;cursor:pointer;">&times;</span>
          </button>
        </li>
      `;
      $(navSelector).append(tabTemplate);
      const paneTemplate = `<div class="tab-pane fade" id="${tabId}" role="tabpanel">${content}</div>`;
      $(contentSelector).append(paneTemplate);
      const newTab = new bootstrap.Tab(document.getElementById(tabId + '-tab'));
      newTab.show();
      initializeTooltips();
      // If this tab is for embeddings visualization, draw the Plotly chart
      if (pageId === "embComp_nlt" || pageId === "embComp_llm" || pageId === "align_nlt" || pageId === "align_llm") {
         drawPlot(pageId);
      }
    }

    // Draw interactive embeddings visualization using Plotly
    function drawPlot(pageId) {
      var dataset;
      var plotTitle = "";
      if (pageId === "embComp_nlt") {
         // Filter llmEmbeddingsData for method "NLT" if available
         dataset = llmEmbeddingsData.filter(function(d){ return d.method && d.method.toUpperCase() === "NLT"; });
         plotTitle = "Embeddings Comparison: NLT to CMT";
      } else if (pageId === "embComp_llm") {
         dataset = llmEmbeddingsData.filter(function(d){ return d.method && d.method.toUpperCase() === "LLM"; });
         plotTitle = "Embeddings Comparison: LLM to CMT";
      } else if (pageId === "align_nlt") {
         dataset = embeddingsData.filter(function(d){ return d.method && d.method.toUpperCase() === "NLT"; });
         plotTitle = "Alignment Results: Aligned NLT to CMT";
      } else if (pageId === "align_llm") {
         dataset = embeddingsData.filter(function(d){ return d.method && d.method.toUpperCase() === "LLM"; });
         plotTitle = "Alignment Results: Aligned LLM to CMT";
      }
      // If filtering yields no rows, use the full dataset as fallback
      if (!dataset || dataset.length === 0) {
         dataset = (pageId.startsWith("embComp_") ? llmEmbeddingsData : embeddingsData);
      }
      // Prepare data arrays for plotting
      var x_mid = [], y_mid = [], hoverTexts = [];
      var arrowShapes = [];
      dataset.forEach(function(d) {
         d.nlt_x = parseFloat(d.nlt_x);
         d.nlt_y = parseFloat(d.nlt_y);
         d.cmt_x = parseFloat(d.cmt_x);
         d.cmt_y = parseFloat(d.cmt_y);
         var midx = (d.nlt_x + d.cmt_x) / 2;
         var midy = (d.nlt_y + d.cmt_y) / 2;
         x_mid.push(midx);
         y_mid.push(midy);
         // Build hover text using model details from csvPagesData
         var model = csvPagesData["page-" + d.key];
         var hoverText = "";
         if (model) {
            hoverText += "<b>" + cleanValue(model.title) + "</b><br/>";
            hoverText += "Key: " + cleanValue(model.key) + "<br/>";
            hoverText += "Theme: " + cleanValue(model.theme) + "<br/>";
            hoverText += "Ontology Type: " + cleanValue(model.ontologyType) + "<br/>";
            hoverText += "Designed For: " + cleanValue(model.designedForTask) + "<br/>";
            hoverText += "Language: " + cleanValue(model.language) + "<br/>";
            hoverText += "Context: " + cleanValue(model.context) + "<br/>";
            hoverText += "Source: " + cleanValue(model.source);
         }
         hoverTexts.push(hoverText);
         // Create an arrow shape from nlt to cmt
         arrowShapes.push({
            type: 'line',
            x0: d.nlt_x,
            y0: d.nlt_y,
            x1: d.cmt_x,
            y1: d.cmt_y,
            line: { color: 'gray', width: 1 },
            xref: 'x', yref: 'y'
         });
      });
      // Create a scatter trace for hover points (midpoints)
      var trace = {
         x: x_mid,
         y: y_mid,
         mode: 'markers',
         type: 'scatter',
         marker: { size: 8, color: 'blue' },
         text: hoverTexts,
         hoverinfo: 'text'
      };
      var dataPlot = [trace];
      var layout = {
         title: plotTitle,
         shapes: arrowShapes,
         dragmode: 'zoom',
         xaxis: { title: 'X' },
         yaxis: { title: 'Y' },
         hovermode: 'closest'
      };
      Plotly.newPlot('plotlyDiv', dataPlot, layout, {responsive: true});
    }

    $(function() {
      // Left sidebar clicks (model details)
      $(".menu-item-left").on("click", function(e) {
        e.preventDefault();
        const title = $(this).data("title");
        const pageId = $(this).data("page");
        const content = renderPageContent(pageId);
        addNewTab(title, content, 'main', pageId);
      });
      // Right sidebar clicks (embeddings visualizations)
      $(".menu-item-right").on("click", function(e) {
        e.preventDefault();
        const title = $(this).data("title");
        const pageId = $(this).data("page");
        const content = renderPageContent(pageId);
        // For right sidebar items, open in the extra tab area
        addNewTab(title, content, 'extra', pageId);
      });
      // Close tab event
      $(document).on('click', '.close-tab', function(e) {
        e.stopPropagation();
        const tabButton = $(this).closest('button');
        const tabId = tabButton.attr('id').replace('-tab','');
        const navContainerId = tabButton.closest('ul.nav-tabs').attr('id');
        if (tabButton.hasClass('active')) {
          tabButton.parent().prev().find('button').trigger('click');
        }
        tabButton.parent().remove();
        $("#" + tabId).remove();
        if (navContainerId === 'mainTab') {
          updateDefaultContentForArea('main');
        } else {
          updateDefaultContentForArea('extra');
        }
      });
      $(document).on('shown.bs.tab', 'button[data-bs-toggle="tab"]', function (e) {
        const tabTitle = $(e.target).text().replace('×','').trim();
        $("#chatContent").html("<p>Chat with <strong>" + tabTitle + "</strong></p>");
      });
      $('.dropdown-submenu > a').on("click", function(e) {
        e.preventDefault();
        e.stopPropagation();
        $(this).next('.dropdown-menu').toggle();
      });
      document.addEventListener('keydown', function(e) {
        if (e.altKey && !e.ctrlKey && !e.shiftKey) {
          let key = e.key.toUpperCase();
          let target = document.querySelector('[data-key="'+key+'"]');
          if (target) {
            target.click();
            e.preventDefault();
          }
        }
      });
      $('#sendChat').on('click', function() {
        const openaiKey = $('#openaiKey').val();
        const query = $('#chatInput').val();
        if (!openaiKey) { alert("Please enter your OpenAI key."); return; }
        if (query.trim() === "") { alert("Please enter a message."); return; }
        $("#chatContent").append("<p><strong>You:</strong> " + query + "</p>");
        $('#chatInput').val('');
      });
      $(document).on('change', '[data-toast-title]', function() {
        var title = $(this).data('toast-title');
        var message = $(this).data('toast-message');
        showToast(title, message);
      });
      
      var originalClasses = $('#lowerTabArea').attr('class');
      $('#toggleExtra').on('change', function() {
        if ($(this).is(':checked')) {
          $('#lowerTabArea').show().addClass(originalClasses);
        } else {
          $('#lowerTabArea').hide().removeClass();
        }
      });
      $('#toggleAssistant').on('change', function() {
        $(this).is(':checked') ? $('#chatPanel').show() : $('#chatPanel').hide();
      });
      $('#toggleLeftPanel').on('change', function() {
        $(this).is(':checked') ? $('#leftPanel').show() : $('#leftPanel').hide();
      });
      $('#toggleRightPanel').on('change', function() {
        $(this).is(':checked') ? $('#rightPanel').show() : $('#rightPanel').hide();
      });
      
      // When clicking on a diagram thumbnail, show the enlarged image in a modal
      $(document).on('click', '.thumbnail-image', function() {
        var src = $(this).attr('src');
        $('#modalImage').attr('src', src);
        var modal = new bootstrap.Modal(document.getElementById('imageModal'));
        modal.show();
      });
    });

    function showToast(title, message) {
      var toastId = 'toast' + Date.now();
      var toastHtml = `<div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
          <div class="toast-header">
            <strong class="me-auto">${title}</strong>
            <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
          </div>
          <div class="toast-body">${message}</div>
        </div>`;
      $('#toastContainer').append(toastHtml);
      var toastElement = document.getElementById(toastId);
      var bsToast = new bootstrap.Toast(toastElement);
      bsToast.show();
      $(toastElement).on('hidden.bs.toast', function () { $(this).remove(); });
    }
  </script>
</body>
</html>
