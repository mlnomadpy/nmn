// Configuration object
const config = {
    line: {
        width: 600,
        height: 450,
        margin: { top: 40, right: 70, bottom: 50, left: 70 },
        colors: {
            dotProduct: '#4a90e2',
            yatProduct: '#e74c3c'
        },
        axisLabels: {
            x: "Angle (degrees)",
            y: "Product Value"
        },
        animation: {
            duration: 300,
            ease: d3.easeCubicOut
        },
        guides: {
            showVerticalGuide: true,
            showIntersectionPoints: true
        }
    },
    scatter: {
        width: 600,
        height: 450,
        margin: { top: 40, right: 40, bottom: 50, left: 40 },
        colors: {
            vector: '#9b59b6',
            reference: '#95a5a6',
            circle: '#bdc3c7'
        },
        axisLabels: {
            x: "X Component",
            y: "Y Component"
        },
        animation: {
            duration: 300,
            ease: d3.easeCubicOut
        },
        guides: {
            showAngleGuide: true,
            showMagnitudeGuide: true,
            showGridLines: true,
            angleArcRadius: 40
        }
    },
    effects: {
        enableParticles: true,
        enable3D: true,
        enableSound: true
    },
    particles: {
        count: 50,
        speed: 0.5,
        color: '#4a90e2',
        size: 2
    },
    theme: {
        light: {
            // ...existing colors...
        },
        dark: {
            background: '#1a1a1a',
            text: '#e0e0e0',
            // ...more color definitions...
        }
    },
    errorHandling: {
        enableLogging: true,
        fallbackValues: {
            vector: [1, 0],
            radius: 100,
            angle: 0
        },
        retryAttempts: 3
    }
};

class ProductVisualization {
    constructor(config) {
        try {
            this.config = config;
            this.elements = {};
            this.errorCount = 0;
            this.initializeVisualization();
        } catch (error) {
            this.handleError('Constructor', error);
        }
    }

    handleError(context, error) {
        const errorMessage = `Error in ${context}: ${error.message}`;
        if (this.config.errorHandling.enableLogging) {
            console.error(errorMessage, error);
        }
        
        // Display user-friendly error message
        const valuesDisplay = document.getElementById("valuesDisplay");
        if (valuesDisplay) {
            valuesDisplay.innerHTML = `
                <div class="error-message">
                    Something went wrong. Please try refreshing the page.
                    ${this.config.errorHandling.enableLogging ? `<br>Error: ${error.message}` : ''}
                </div>
            `;
        }

        // Attempt recovery if possible
        this.attemptRecovery(context);
    }

    attemptRecovery(context) {
        try {
            switch (context) {
                case 'updateProducts':
                    if (this.errorCount < this.config.errorHandling.retryAttempts) {
                        this.errorCount++;
                        const [x, y] = this.config.errorHandling.fallbackValues.vector;
                        setTimeout(() => this.updateProducts(x, y), 1000);
                    }
                    break;
                // Add more recovery cases as needed
            }
        } catch (error) {
            console.error('Recovery attempt failed:', error);
        }
    }

    initializeVisualization() {
        try {
            this.createLineChart();
            this.createScatterPlot();
            this.initializeControls();
            this.setupDragBehavior();
            
            // Initial update
            this.updateProducts(
                this.config.scatter.width/2 + 100,
                this.config.scatter.height/2
            );
        } catch (error) {
            this.handleError('initializeVisualization', error);
        }
    }

    createLineChart() {
        try {
            const { width, height, margin, colors } = this.config.line;
            
            // Create SVG
            this.elements.lineSvg = d3.select("#lineChart")
                .append("svg")
                .attr("viewBox", `0 0 ${width} ${height}`)
                .attr("preserveAspectRatio", "xMidYMid meet");

            // Create scales
            this.elements.xScale = d3.scaleLinear()
                .domain([0, 360])
                .range([margin.left, width - margin.right]);

            this.elements.yScale = d3.scaleLinear()
                .range([height - margin.bottom, margin.top]);

            // Add axes and grid
            this.setupLineChartAxes();
            this.setupLineChartGrid();

            // Create paths for lines
            this.elements.dotPath = this.elements.lineSvg.append("path")
                .attr("fill", "none")
                .attr("stroke", colors.dotProduct)
                .attr("stroke-width", 2.5);

            this.elements.yatPath = this.elements.lineSvg.append("path")
                .attr("fill", "none")
                .attr("stroke", colors.yatProduct)
                .attr("stroke-width", 2.5);

            // Add highlight points
            this.elements.dotHighlight = this.elements.lineSvg.append("circle")
                .attr("class", "highlight-point")
                .attr("r", 0)
                .attr("fill", colors.dotProduct);

            this.elements.yatHighlight = this.elements.lineSvg.append("circle")
                .attr("class", "highlight-point")
                .attr("r", 0)
                .attr("fill", colors.yatProduct);

            this.addLineChartLegend();
        } catch (error) {
            this.handleError('createLineChart', error);
        }
    }

    createScatterPlot() {
        try {
            const { width, height, margin, colors } = this.config.scatter;
            
            // Create SVG
            this.elements.scatterSvg = d3.select("#scatterPlot")
                .append("svg")
                .attr("viewBox", `0 0 ${width} ${height}`)
                .attr("preserveAspectRatio", "xMidYMid meet");

            // Create scales
            this.elements.scatterXScale = d3.scaleLinear()
                .domain([-2, 2])
                .range([margin.left, width - margin.right]);

            this.elements.scatterYScale = d3.scaleLinear()
                .domain([-2, 2])
                .range([height - margin.bottom, margin.top]);

            // Add axes and grid
            this.setupScatterPlotAxes();
            this.setupScatterPlotGrid();

            // Add circle elements
            this.addScatterPlotElements();
        } catch (error) {
            this.handleError('createScatterPlot', error);
        }
    }

    setupLineChartAxes() {
        try {
            const { width, height, margin } = this.config.line;

            // X-axis
            this.elements.lineSvg.append("g")
                .attr("transform", `translate(0,${height - margin.bottom})`)
                .call(d3.axisBottom(this.elements.xScale)
                    .ticks(8)
                    .tickFormat(d => d + "°"));

            // Y-axis
            this.elements.yAxisGroup = this.elements.lineSvg.append("g")
                .attr("transform", `translate(${margin.left},0)`);

            // Add axis labels
            this.elements.lineSvg.append("text")
                .attr("class", "axis-label")
                .attr("x", width / 2)
                .attr("y", height - 10)
                .attr("text-anchor", "middle")
                .text(this.config.line.axisLabels.x);

            this.elements.lineSvg.append("text")
                .attr("class", "axis-label")
                .attr("transform", "rotate(-90)")
                .attr("x", -height / 2)
                .attr("y", 15)
                .attr("text-anchor", "middle")
                .text(this.config.line.axisLabels.y);
        } catch (error) {
            this.handleError('setupLineChartAxes', error);
        }
    }

    setupLineChartGrid() {
        try {
            const { width, height, margin } = this.config.line;

            // X-grid
            this.elements.lineSvg.append("g")
                .attr("class", "grid")
                .attr("opacity", 0.1)
                .call(d3.axisBottom(this.elements.xScale)
                    .ticks(8)
                    .tickSize(height - margin.top - margin.bottom)
                    .tickFormat(""));

            // Y-grid
            this.elements.yGridGroup = this.elements.lineSvg.append("g")
                .attr("class", "grid")
                .attr("opacity", 0.1)
                .attr("transform", `translate(${margin.left},0)`);
        } catch (error) {
            this.handleError('setupLineChartGrid', error);
        }
    }

    addLineChartLegend() {
        try {
            const { width, margin, colors } = this.config.line;
            
            const legend = this.elements.lineSvg.append("g")
                .attr("transform", `translate(${width - margin.right + 10},${margin.top})`);

            // Dot product legend
            legend.append("line")
                .attr("x1", 0)
                .attr("x2", 20)
                .attr("stroke", colors.dotProduct)
                .attr("stroke-width", 2);

            legend.append("text")
                .attr("x", 25)
                .attr("dy", "0.32em")
                .text("Dot Product");

            // Yat product legend
            legend.append("line")
                .attr("x1", 0)
                .attr("x2", 20)
                .attr("y1", 20)
                .attr("y2", 20)
                .attr("stroke", colors.yatProduct)
                .attr("stroke-width", 2);

            legend.append("text")
                .attr("x", 25)
                .attr("y", 20)
                .attr("dy", "0.32em")
                .text("Yat Product");
        } catch (error) {
            this.handleError('addLineChartLegend', error);
        }
    }

    setupScatterPlotAxes() {
        try {
            const { width, height } = this.config.scatter;

            this.elements.scatterSvg.append("g")
                .attr("transform", `translate(0,${height/2})`)
                .call(d3.axisBottom(this.elements.scatterXScale));

            this.elements.scatterSvg.append("g")
                .attr("transform", `translate(${width/2},0)`)
                .call(d3.axisLeft(this.elements.scatterYScale));

            // Add axis labels
            this.elements.scatterSvg.append("text")
                .attr("class", "axis-label")
                .attr("x", width / 2)
                .attr("y", height - 10)
                .attr("text-anchor", "middle")
                .text(this.config.scatter.axisLabels.x);

            this.elements.scatterSvg.append("text")
                .attr("class", "axis-label")
                .attr("transform", "rotate(-90)")
                .attr("x", -height / 2)
                .attr("y", 15)
                .attr("text-anchor", "middle")
                .text(this.config.scatter.axisLabels.y);
        } catch (error) {
            this.handleError('setupScatterPlotAxes', error);
        }
    }

    setupScatterPlotGrid() {
        try {
            const { width, height, margin } = this.config.scatter;

            this.elements.scatterSvg.append("g")
                .attr("class", "grid")
                .attr("opacity", 0.1)
                .call(d3.axisBottom(this.elements.scatterXScale)
                    .tickSize(height - margin.top - margin.bottom)
                    .tickFormat(""));

            this.elements.scatterSvg.append("g")
                .attr("class", "grid")
                .attr("opacity", 0.1)
                .call(d3.axisLeft(this.elements.scatterYScale)
                    .tickSize(-width + margin.left + margin.right)
                    .tickFormat(""));
        } catch (error) {
            this.handleError('setupScatterPlotGrid', error);
        }
    }

    addScatterPlotElements() {
        try {
            const { width, height, colors } = this.config.scatter;

            // Add coordinate grid lines
            if (this.config.scatter.guides.showGridLines) {
                this.elements.scatterSvg.selectAll('.coordinate-line')
                    .data([-1, 0, 1])
                    .enter()
                    .append('line')
                    .attr('class', 'coordinate-line')
                    .attr('x1', d => this.elements.scatterXScale(d))
                    .attr('x2', d => this.elements.scatterXScale(d))
                    .attr('y1', this.elements.scatterYScale(-2))
                    .attr('y2', this.elements.scatterYScale(2))
                    .attr('stroke', '#ddd')
                    .attr('stroke-width', 1)
                    .attr('stroke-dasharray', '4');

                this.elements.scatterSvg.selectAll('.coordinate-line-h')
                    .data([-1, 0, 1])
                    .enter()
                    .append('line')
                    .attr('class', 'coordinate-line')
                    .attr('y1', d => this.elements.scatterYScale(d))
                    .attr('y2', d => this.elements.scatterYScale(d))
                    .attr('x1', this.elements.scatterXScale(-2))
                    .attr('x2', this.elements.scatterXScale(2))
                    .attr('stroke', '#ddd')
                    .attr('stroke-width', 1)
                    .attr('stroke-dasharray', '4');
            }

            // Add unit circle
            this.elements.scatterSvg.append("circle")
                .attr("cx", width/2)
                .attr("cy", height/2)
                .attr("r", Math.abs(this.elements.scatterXScale(1) - this.elements.scatterXScale(0)))
                .attr("fill", "none")
                .attr("stroke", colors.circle)
                .attr("stroke-width", 1)
                .attr("stroke-dasharray", "4");

            // Add reference vector
            this.elements.refVector = this.elements.scatterSvg.append("line")
                .attr("x1", width/2)
                .attr("y1", height/2)
                .attr("x2", this.elements.scatterXScale(1))
                .attr("y2", height/2)
                .attr("stroke", colors.reference)
                .attr("stroke-width", 2)
                .attr("stroke-dasharray", "4");

            // Add adjustable circle
            this.elements.circle = this.elements.scatterSvg.append("circle")
                .attr("cx", width/2)
                .attr("cy", height/2)
                .attr("r", 100)
                .attr("fill", "none")
                .attr("stroke", colors.circle)
                .attr("stroke-width", 1.5)
                .attr("stroke-dasharray", "4");

            // Add vector line
            this.elements.vectorLine = this.elements.scatterSvg.append("line")
                .attr("x1", width/2)
                .attr("y1", height/2)
                .attr("x2", width/2 + 100)
                .attr("y2", height/2)
                .attr("stroke", colors.vector)
                .attr("stroke-width", 2.5);

            // Add angle arc
            this.elements.angleArc = this.elements.scatterSvg.append("path")
                .attr("fill", "none")
                .attr("stroke", colors.reference)
                .attr("stroke-width", 1.5);

            // Add draggable point
            this.elements.point = this.elements.scatterSvg.append("circle")
                .attr("cx", width/2 + 100)
                .attr("cy", height/2)
                .attr("r", 8)
                .attr("fill", colors.vector)
                .attr("stroke", "white")
                .attr("stroke-width", 2)
                .style("cursor", "move")
                .style("filter", "drop-shadow(0 2px 2px rgba(0,0,0,0.2))");

            // Add angle text
            this.elements.angleText = this.elements.scatterSvg.append("text")
                .attr("class", "vector-element")
                .attr("fill", colors.reference)
                .attr("font-size", "12px")
                .attr("text-anchor", "middle")
                .attr("opacity", 0);

            // Add magnitude guide text
            this.elements.magnitudeText = this.elements.scatterSvg.append("text")
                .attr("class", "vector-element")
                .attr("fill", colors.vector)
                .attr("font-size", "12px")
                .attr("text-anchor", "start")
                .attr("opacity", 0);
        } catch (error) {
            this.handleError('addScatterPlotElements', error);
        }
    }

    initializeControls() {
        try {
            const radiusSlider = document.getElementById("radiusSlider");
            const radiusValue = document.getElementById("radiusValue");

            radiusSlider.addEventListener("input", (event) => {
                const value = event.target.value;
                this.elements.circle.attr("r", value);
                radiusValue.textContent = value;
            });
        } catch (error) {
            this.handleError('initializeControls', error);
        }
    }

    setupDragBehavior() {
        try {
            const { width, height, margin } = this.config.scatter;

            // Create tooltip
            this.elements.tooltip = d3.select("body")
                .append("div")
                .attr("class", "tooltip")
                .style("opacity", 0);

            // Enhanced tooltip content
            const formatTooltip = (x, y) => `
                <span class="tooltip-label">Coordinates</span>
                <span class="tooltip-value">(${x}, ${y})</span>
                <span class="tooltip-label">Magnitude</span>
                <span class="tooltip-value">${Math.sqrt(x*x + y*y).toFixed(2)}</span>
            `;

            // Create drag behavior
            const drag = d3.drag()
                .on("start", (event) => {
                    d3.select(event.sourceEvent.target)
                        .attr("r", 10)
                        .style("filter", "drop-shadow(0 3px 3px rgba(0,0,0,0.3))");
                })
                .on("drag", (event) => {
                    const x = Math.max(margin.left, Math.min(width - margin.right, event.x));
                    const y = Math.max(margin.top, Math.min(height - margin.bottom, event.y));

                    d3.select(event.sourceEvent.target)
                        .attr("cx", x)
                        .attr("cy", y);

                    this.updateProducts(x, y);

                    const xVal = this.elements.scatterXScale.invert(x).toFixed(2);
                    const yVal = this.elements.scatterYScale.invert(y).toFixed(2);

                    this.elements.tooltip
                        .style("opacity", 1)
                        .html(formatTooltip(xVal, yVal))
                        .style("left", (event.sourceEvent.pageX + 10) + "px")
                        .style("top", (event.sourceEvent.pageY - 10) + "px");
                })
                .on("end", (event) => {
                    d3.select(event.sourceEvent.target)
                        .attr("r", 8)
                        .style("filter", "drop-shadow(0 2px 2px rgba(0,0,0,0.2))");
                    this.elements.tooltip.style("opacity", 0);
                });

            this.elements.point.call(drag);
        } catch (error) {
            this.handleError('setupDragBehavior', error);
            // Disable drag functionality if it fails
            this.elements.point?.style('pointer-events', 'none');
        }
    }

    calculateProducts(vector) {
        try {
            const angles = d3.range(0, 361, 1);
            return angles.map(angle => {
                const radians = (angle * Math.PI) / 180;
                const unitVector = [Math.cos(radians), Math.sin(radians)];
                
                const dotProduct = unitVector[0] * vector[0] + unitVector[1] * vector[1];
                const euclidean_distance = (unitVector[0] - vector[0]) ** 2 + (unitVector[1] - vector[1]) ** 2 + 1e-7;
                const yatProduct = (dotProduct * dotProduct) / (euclidean_distance * euclidean_distance);
                
                return {
                    angle,
                    dotProduct: dotProduct,
                    yatProduct
                };
            });
        } catch (error) {
            this.handleError('calculateProducts', error);
            // Return fallback data
            return Array.from({length: 361}, (_, i) => ({
                angle: i,
                dotProduct: 0,
                yatProduct: 0
            }));
        }
    }

    updateProducts(x, y) {
        try {
            const { width, height } = this.config.scatter;
            
            // Convert coordinates to vector
            const vector = [
                this.elements.scatterXScale.invert(x),
                this.elements.scatterYScale.invert(y)
            ];

            // Calculate angle and update arc
            const angle = Math.atan2(vector[1], vector[0]);
            const normalizedAngle = angle < 0 ? angle + 2 * Math.PI : angle;
            const degrees = (normalizedAngle * 180) / Math.PI;

            // Update vector line with animation
            this.elements.vectorLine
                .transition()
                .duration(this.config.scatter.animation.duration)
                .ease(this.config.scatter.animation.ease)
                .attr("x2", x)
                .attr("y2", y);

            // Update angle arc and text
            this.updateAngleArc(normalizedAngle);
            const arcCenterX = width/2 + this.config.scatter.guides.angleArcRadius/2;
            const arcCenterY = height/2 - this.config.scatter.guides.angleArcRadius/2;
            
            this.elements.angleText
                .attr("x", arcCenterX)
                .attr("y", arcCenterY)
                .attr("opacity", 1)
                .text(`${Math.round(degrees)}°`);

            // Update magnitude guide
            if (this.config.scatter.guides.showMagnitudeGuide) {
                const magnitude = Math.sqrt(vector[0] * vector[0] + vector[1] * vector[1]);
                const midX = (width/2 + x) / 2;
                const midY = (height/2 + y) / 2;
                
                this.elements.magnitudeText
                    .attr("x", midX + 5)
                    .attr("y", midY - 5)
                    .attr("opacity", 1)
                    .text(`|v| = ${magnitude.toFixed(2)}`);
            }

            this.updateAngleArc(normalizedAngle);
            this.updateLineChart(this.calculateProducts(vector));
            
            // Update vector line with animation
            this.elements.vectorLine
                .transition()
                .duration(this.config.scatter.animation.duration)
                .ease(this.config.scatter.animation.ease)
                .attr("x2", x)
                .attr("y2", y);

            // Update magnitude guide if enabled
            if (this.config.scatter.guides.showMagnitudeGuide && this.elements.magnitudeText) {
                const magnitude = Math.sqrt(vector[0] * vector[0] + vector[1] * vector[1]);
                this.elements.magnitudeText
                    .attr("x", x + 10)
                    .attr("y", y - 10)
                    .attr("opacity", 1)
                    .text(`|v| = ${magnitude.toFixed(2)}`);
            }

            this.updateValuesDisplay(vector, degrees);
        } catch (error) {
            this.handleError('updateProducts', error);
            // Use fallback values
            const fallbackX = this.config.scatter.width/2;
            const fallbackY = this.config.scatter.height/2;
            if (x !== fallbackX && y !== fallbackY) {
                this.updateProducts(fallbackX, fallbackY);
            }
        }
    }

    updateAngleArc(angle) {
        try {
            const { width, height } = this.config.scatter;
            const radius = this.config.scatter.guides.angleArcRadius;

            // Fix arc generation
            const arcGenerator = d3.arc()
                .innerRadius(0)
                .outerRadius(radius)
                .startAngle(0)
                .endAngle(angle)
                .cornerRadius(0);  // Ensure sharp corners

            // Create arc path
            const arcPath = arcGenerator({});

            this.elements.angleArc
                .transition()
                .duration(this.config.scatter.animation.duration)
                .ease(this.config.scatter.animation.ease)
                .attr("d", arcPath)
                .attr("transform", `translate(${width/2},${height/2})`);

            // Update angle text position
            if (this.elements.angleText) {
                const textRadius = radius * 0.7;  // Position text inside the arc
                const textAngle = angle / 2;  // Position text at middle of arc
                const textX = Math.cos(textAngle) * textRadius;
                const textY = -Math.sin(textAngle) * textRadius;  // Negative for SVG coordinates

                this.elements.angleText
                    .attr("transform", `translate(${width/2},${height/2})`)
                    .attr("x", textX)
                    .attr("y", textY)
                    .attr("opacity", 1)
                    .text(`${Math.round((angle * 180) / Math.PI)}°`);
            }
        } catch (error) {
            this.handleError('updateAngleArc', error);
        }
    }

    updateLineChart(data) {
        try {
            // Update y-scale based on data range
            const yExtent = d3.extent([...data.map(d => d.dotProduct), ...data.map(d => d.yatProduct)]);
            const yPadding = (yExtent[1] - yExtent[0]) * 0.1;
            this.elements.yScale.domain([yExtent[0] - yPadding, yExtent[1] + yPadding]);

            // Update axes and grid
            this.elements.yAxisGroup
                .transition()
                .duration(300)
                .call(d3.axisLeft(this.elements.yScale));

            this.elements.yGridGroup
                .transition()
                .duration(300)
                .call(d3.axisLeft(this.elements.yScale)
                    .tickSize(-this.config.line.width + this.config.line.margin.left + this.config.line.margin.right)
                    .tickFormat(""));

            // Create line generators
            const lineGenerator = d3.line()
                .x(d => this.elements.xScale(d.angle))
                .y(d => this.elements.yScale(d.dotProduct))
                .curve(d3.curveCatmullRom);

            const yatLineGenerator = d3.line()
                .x(d => this.elements.xScale(d.angle))
                .y(d => this.elements.yScale(d.yatProduct))
                .curve(d3.curveCatmullRom);

            // Update paths
            this.elements.dotPath
                .datum(data)
                .transition()
                .duration(this.config.line.animation.duration)
                .ease(this.config.line.animation.ease)
                .attr("d", lineGenerator);

            this.elements.yatPath
                .datum(data)
                .transition()
                .duration(300)
                .attr("d", yatLineGenerator);

            // Add vertical guide line
            if (this.config.line.guides.showVerticalGuide) {
                const guide = this.elements.lineSvg.selectAll('.vertical-guide')
                    .data([data[0].angle]);

                guide.enter()
                    .append('line')
                    .attr('class', 'vertical-guide vector-element')
                    .merge(guide)
                    .attr('x1', d => this.elements.xScale(d))
                    .attr('x2', d => this.elements.xScale(d))
                    .attr('y1', this.config.line.margin.top)
                    .attr('y2', this.config.line.height - this.config.line.margin.bottom)
                    .attr('stroke', '#aaa')
                    .attr('stroke-width', 1)
                    .attr('stroke-dasharray', '3,3');
            }

            // Update highlights
            const angleIndex = Math.round(data[0].angle);
            const dataPoint = data[angleIndex];

            this.elements.dotHighlight
                .attr("cx", this.elements.xScale(angleIndex))
                .attr("cy", this.elements.yScale(dataPoint.dotProduct))
                .attr("r", 6);

            this.elements.yatHighlight
                .attr("cx", this.elements.xScale(angleIndex))
                .attr("cy", this.elements.yScale(dataPoint.yatProduct))
                .attr("r", 6);
        } catch (error) {
            this.handleError('updateLineChart', error);
        }
    }

    updateValuesDisplay(vector, degrees) {
        try {
            const magnitude = Math.sqrt(vector[0] * vector[0] + vector[1] * vector[1]);
            const dotProduct = vector[0] * 1 + vector[1] * 0;
            const normalizedDotProduct = dotProduct / magnitude;
            const yatProduct = (dotProduct * dotProduct) / (magnitude * magnitude);

            document.getElementById("valuesDisplay").innerHTML = `
                <div class="values-grid">
                    <strong>Vector:</strong>
                    <span>(${vector[0].toFixed(3)}, ${vector[1].toFixed(3)})</span>
                    <strong>Magnitude:</strong>
                    <span>${magnitude.toFixed(3)}</span>
                    <strong>Angle:</strong>
                    <span>${degrees.toFixed(1)}°</span>
                    <strong>Dot Product:</strong>
                    <span>${normalizedDotProduct.toFixed(3)}</span>
                    <strong>Yat Product:</strong>
                    <span>${yatProduct.toFixed(3)}</span>
                </div>
            `;
        } catch (error) {
            this.handleError('updateValuesDisplay', error);
        }
    }

    initializeParticleSystem() {
        try {
            if (!this.config.effects.enableParticles) return;

            this.particles = [];
            const particleGroup = this.elements.scatterSvg.append('g')
                .attr('class', 'particles');

            for (let i = 0; i < this.config.particles.count; i++) {
                this.particles.push({
                    x: Math.random() * this.config.scatter.width,
                    y: Math.random() * this.config.scatter.height,
                    vx: (Math.random() - 0.5) * this.config.particles.speed,
                    vy: (Math.random() - 0.5) * this.config.particles.speed
                });
            }

            this.updateParticles();
        } catch (error) {
            this.handleError('initializeParticleSystem', error);
            this.config.effects.enableParticles = false; // Disable particles on error
        }
    }

    updateParticles() {
        try {
            const particleGroup = this.elements.scatterSvg.select('.particles');
            const particles = particleGroup.selectAll('circle')
                .data(this.particles);

            particles.enter()
                .append('circle')
                .attr('r', this.config.particles.size)
                .attr('fill', this.config.particles.color)
                .merge(particles)
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);

            particles.exit().remove();

            this.particles.forEach(p => {
                p.x += p.vx;
                p.y += p.vy;

                if (p.x < 0 || p.x > this.config.scatter.width) p.vx *= -1;
                if (p.y < 0 || p.y > this.config.scatter.height) p.vy *= -1;
            });

            requestAnimationFrame(this.updateParticles.bind(this));
        } catch (error) {
            this.handleError('updateParticles', error);
        }
    }

    setup3DEffects() {
        try {
            if (!this.config.effects.enable3D) return;

            // Implement 3D effects setup here
        } catch (error) {
            this.handleError('setup3DEffects', error);
        }
    }

    initializeAudioFeedback() {
        try {
            if (!this.config.effects.enableSound) return;

            // Implement audio feedback setup here
        } catch (error) {
            this.handleError('initializeAudioFeedback', error);
        }
    }
}

// Add global error handling
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    document.getElementById('valuesDisplay').innerHTML = `
        <div class="error-message">
            An unexpected error occurred. Please refresh the page.
        </div>
    `;
});

// Initialize visualization when document is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        if (!d3) throw new Error('D3.js library not loaded');
        new ProductVisualization(config);
    } catch (error) {
        console.error('Fatal error during initialization:', error);
        document.getElementById('valuesDisplay').innerHTML = `
            <div class="error-message">
                Could not initialize visualization.
                <br>
                Error: ${error.message}
                <br>
                Please ensure all required libraries are loaded and try refreshing the page.
            </div>
        `;
    }
});