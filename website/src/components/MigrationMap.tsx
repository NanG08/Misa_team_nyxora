import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { MigrationPoint } from '../types';
import { motion, AnimatePresence } from 'motion/react';

interface MigrationMapProps {
  points: MigrationPoint[];
  selectedSpecies?: string;
  currentTime?: string; // ISO date string
  showEnvironmentalOverlay?: 'none' | 'temperature' | 'forest' | 'drought';
}

export const MigrationMap: React.FC<MigrationMapProps> = ({ 
  points, 
  selectedSpecies, 
  currentTime,
  showEnvironmentalOverlay = 'none'
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [tooltip, setTooltip] = useState<{ point: MigrationPoint; x: number; y: number } | null>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const projection = d3.geoMercator()
      .scale(width / 2 / Math.PI)
      .translate([width / 2, height / 1.5]);

    const path = d3.geoPath().projection(projection);

    // Draw world map
    d3.json('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson').then((data: any) => {
      // Base Map
      svg.append('g')
        .selectAll('path')
        .data(data.features)
        .enter()
        .append('path')
        .attr('d', path)
        .attr('fill', '#e5e5e5')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 0.5);

      // Environmental Overlay (Mocked with random heatmaps for demo)
      if (showEnvironmentalOverlay !== 'none') {
        const overlayColor = {
          temperature: '#ff4e00',
          forest: '#10b981',
          drought: '#f59e0b'
        }[showEnvironmentalOverlay];

        svg.append('g')
          .selectAll('circle')
          .data(d3.range(50))
          .enter()
          .append('circle')
          .attr('cx', () => Math.random() * width)
          .attr('cy', () => Math.random() * height)
          .attr('r', () => Math.random() * 100 + 50)
          .attr('fill', overlayColor)
          .attr('opacity', 0.1)
          .style('filter', 'blur(20px)');
      }

      // Filter points by species and time
      let filteredPoints = selectedSpecies 
        ? points.filter(p => p.species === selectedSpecies)
        : points;

      if (currentTime) {
        filteredPoints = filteredPoints.filter(p => p.timestamp <= currentTime);
      }

      // Draw migration paths
      const speciesGroups = d3.group(filteredPoints, d => d.species);
      
      speciesGroups.forEach((group, species) => {
        const line = d3.line<MigrationPoint>()
          .x(d => projection([d.lng, d.lat])![0])
          .y(d => projection([d.lng, d.lat])![1])
          .curve(d3.curveCatmullRom.alpha(0.5));

        svg.append('path')
          .datum(group)
          .attr('fill', 'none')
          .attr('stroke', species === 'Monarch Butterfly' ? '#F27D26' : species === 'Arctic Tern' ? '#3b82f6' : '#8b5cf6')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5')
          .attr('d', line);
      });

      // Draw points with animation and interaction
      const pointsGroup = svg.selectAll('.point')
        .data(filteredPoints, (d: any) => `${d.species}-${d.timestamp}`);

      pointsGroup.exit()
        .transition()
        .duration(300)
        .attr('r', 0)
        .remove();

      const pointsEnter = pointsGroup.enter()
        .append('circle')
        .attr('class', 'point')
        .attr('cx', d => projection([d.lng, d.lat])![0])
        .attr('cy', d => projection([d.lng, d.lat])![1])
        .attr('r', 0)
        .attr('fill', d => d.species === 'Monarch Butterfly' ? '#F27D26' : d.species === 'Arctic Tern' ? '#3b82f6' : '#8b5cf6')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer');

      pointsEnter.merge(pointsGroup as any)
        .transition()
        .duration(500)
        .attr('r', 4)
        .attr('opacity', d => d.timestamp === currentTime ? 1 : 0.4);

      pointsEnter.merge(pointsGroup as any)
        .on('mouseenter', function(event: MouseEvent, d: MigrationPoint) {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', 8)
            .attr('stroke-width', 2);
          
          const rect = containerRef.current?.getBoundingClientRect();
          if (rect) {
            setTooltip({
              point: d,
              x: event.clientX - rect.left,
              y: event.clientY - rect.top
            });
          }
        })
        .on('mouseleave', function() {
          d3.select(this)
            .transition()
            .duration(200)
            .attr('r', 4)
            .attr('stroke-width', 1);
          setTooltip(null);
        });
    });

    return () => {
      setTooltip(null);
    };

  }, [points, selectedSpecies, currentTime, showEnvironmentalOverlay]);

  return (
    <div ref={containerRef} className="w-full h-full relative bg-[#f0f0f0] rounded-xl overflow-hidden border border-line shadow-inner">
      <svg ref={svgRef} className="w-full h-full" />
      
      <AnimatePresence>
        {tooltip && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.15 }}
            className="absolute pointer-events-none z-50"
            style={{
              left: tooltip.x + 15,
              top: tooltip.y - 10,
            }}
          >
            <div className="bg-ink text-bg p-4 rounded-xl shadow-2xl border border-white/10 min-w-[200px]">
              <div className="text-[10px] font-mono uppercase tracking-wider opacity-50 mb-2">Migration Point</div>
              <div className="text-sm font-bold mb-3">{tooltip.point.species}</div>
              <div className="space-y-1.5 text-xs">
                <div className="flex justify-between">
                  <span className="opacity-60">Timestamp:</span>
                  <span className="font-mono">{tooltip.point.timestamp}</span>
                </div>
                <div className="flex justify-between">
                  <span className="opacity-60">Status:</span>
                  <span className="font-mono capitalize">{tooltip.point.status}</span>
                </div>
                <div className="flex justify-between">
                  <span className="opacity-60">Location:</span>
                  <span className="font-mono">{tooltip.point.lat.toFixed(2)}°, {tooltip.point.lng.toFixed(2)}°</span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="absolute bottom-4 left-4 bg-white/80 backdrop-blur-md p-3 rounded-lg border border-line text-[10px] font-mono uppercase tracking-wider">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-2 h-2 rounded-full bg-[#F27D26]" />
          <span>Monarch Butterfly</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#3b82f6]" />
          <span>Arctic Tern</span>
        </div>
      </div>
    </div>
  );
};
