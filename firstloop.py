#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 12:37:31 2025

@author: qeiminipc
"""

"""

This code forms a very basic control for a power meter.

!!! BEFORE RUNNING THE CODE !!!
(1) Turn on all instruments (rotating stages, powermeter, laser)
(2) Change file names / directories if needed
(3) Change addresses for the rotating stages (use lsusb and search for Future Technology Devices International)

"""

import pyvisa
import pyvisa.errors
import usb
import time
import numpy as np
from scipy.optimize import curve_fit
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


rm = pyvisa.ResourceManager()

pmAddr = 'USB0::4883::32886::M01112547::0::INSTR' #Must be adjusted

"""
USB0:: Indicates the connection is via USB.
4883:: This part likely represents the Vendor ID (VID) of the instrument's manufacturer. It uniquely identifies the company that produced the device.
32885:: This part is likely the Product ID (PID). It identifies the specific type of device from that vendor.
P5002859:: This segment seems to represent the serial number or a unique identifier assigned to the specific instrument.
0:: This might represent the USB interface number or a specific endpoint within the device, although there can be variations in how this is included.
INSTR: This signifies that the device is a general instrument that can be controlled via the VISA library. 
"""

from motor_controller import MotorController

stage = MotorController("127.0.0.1","55000")

# Function to setup the power meter instrument with error handling
def setup_powermeter(addr,countNum):
    try:
        # Open the resource
        pmeter = rm.open_resource(addr, open_timeout=1000)
        
        # Set the communication timeout, taking integration time into account
        pmeter.timeout = 5000 + 0.5*countNum
        
        # Query and print the instrument's identification
        print(pmeter.query("*IDN?"))
        
        # Configure the instrument settings
        pmeter.write("SENS:CURR:RANGE:AUTO ON")  # Set the range to auto
        pmeter.write("SENS:CORR:WAV 1550")  # Set the wavelength to 1550 nm
        pmeter.write("SENS:POW:UNIT W")     # Set the power unit to watts
        pmeter.write("sense:average:count "+str(int(countNum)))      # Set the averaging to 1000
        
        # Return the instrument object
        print("Powermeter has been succesfully set up.")
        return pmeter

    except pyvisa.errors.VisaIOError as e:
        print(f"Error setting up instrument: {str(e)}")
        return e
    
    except usb.core.USBTimeoutError as e:
        print(f'usb error: {str(e)}')
        return e
    except:
        print('unknown error on this pmeter:')
        print(pmAddr)
        return -1
    
def getPower(pm):
    start = time.time()
    try:
        pow = float(pm.query("meas:pow?"))
        end = time.time()
        return pow, start, end

    except pyvisa.errors.VisaIOError as e:
        raise pyvisa.errors.VisaIOError
        
# spin a stage and measure power from a powermeter

def setCountTime(pm, countNum):
        pm.write("sense:average:count "+str(int(countNum)))
        pm.timeout = 5000 + 0.5*countNum

def rotateAndCount(stage,start,end,stepSize,pm,countNum):
    angles = np.arange(start,end,stepSize)
    nAngles = np.size(angles)
    powers = np.zeros((nAngles,3))

    setCountTime(pm, countNum)

    for n in np.arange(nAngles):

        curPos = float(stage.getPos('TestELL16'))

        target_angle = angles[n] % 360

        # Calculate the difference
        delta = target_angle - curPos

        # Adjust the delta to find the shortest path
        if delta > 180:
            delta -= 360  # Move counter-clockwise (negative)
        elif delta < -180:
            delta += 360  # Move clockwise (positive)

        # Now, 'delta' contains the shortest distance and direction
        print(f'Moving from {curPos} to {target_angle} by {delta} degrees')
        stage.move('TestELL16', delta)


        time.sleep(0.100)

        power = getPower(pm)
        stgAngle = stage.getPos('TestELL16')

        powers[n,0] = angles[n]
        powers[n,1] = stgAngle
        powers[n,2] = power[0]
        print('power', power[0])

    return powers


def measZerosLoop(fname,stage,zero1,zero2,range,stepSize,pm,countNum,nLoops):
    stage.home('TestELL16')
    for n in np.arange(nLoops):
        #stage.home('TestELL16')


        data1 = rotateAndCount(stage,zero1-range,zero1+range,stepSize,pm,countNum)
        data2 = rotateAndCount(stage,zero2-range,zero2+range,stepSize,pm,countNum)
    
        data  =  np.vstack((data1,data2))

        if n == 0:
            np.savetxt(fname,data,delimiter=',')
            dataNew = data
        else:
            dataSaved = np.genfromtxt(fname,delimiter=',')
            dataNew = np.hstack((dataSaved,data[:,1:]))
            np.savetxt(fname,dataNew,delimiter=',')
         
    return dataNew        
        
def measFlickerLoop(fname,stage,zero1,zero2,pm,countNum,nLoops):
    """     Attempts to rotate forward and backwards between the two given positions, measuring position and power.
            Make sure to set the countNum high enough to be below the recommended duty cycle: 40%"""

    setCountTime(pm, countNum)

    stage.home('TestELL16')
    stage.goto('TestELL16',zero1)
    distance = zero2-zero1

    powers = np.zeros((1,4))
    dataNew = np.array([])

    for n in np.arange(nLoops):
        time.sleep(0.200)

        power1 = getPower(pm)
        stgAngle1 = float(stage.getAPos('TestELL16'))

        powers[0,0] = stgAngle1
        powers[0,1] = power1[0]
        print(f"Loop {n+1}/{nLoops} | Pos1: {stgAngle1:.4f}, Power1: {power1[0]:.4f}")

        stage.forward('TestELL16',distance)

        time.sleep(0.200)

        power2 = getPower(pm)
        stgAngle2 = float(stage.getAPos('TestELL16'))

        powers[0,2] = stgAngle2
        powers[0,3] = power2[0]
        print(f"Loop {n+1}/{nLoops} | Pos2: {stgAngle2:.4f}, Power2: {power2[0]:.4f}")

        stage.backward('TestELL16',distance)

        if n == 0:
            np.savetxt(fname,powers,delimiter=',')
            dataNew = powers
        else:
            dataSaved = np.genfromtxt(fname,delimiter=',')
            # Reshape is needed if only one row was saved previously
            if dataSaved.ndim == 1:
                dataSaved = dataSaved.reshape(1, -1)
            dataNew = np.vstack((dataSaved,powers))
            np.savetxt(fname,dataNew,delimiter=',')

    return dataNew

def analyseFlickerLoop(fname, zero1, zero2):
    """
    Generates four subplots to analyze the flicker measurement data.
    1. Histogram of position errors at zero1.
    2. Scatter plot and linear regression of power at zero1.
    3. Histogram of position errors at zero2.
    4. Scatter plot and linear regression of power at zero2.
    """
    # --- Data Extraction ---
    data = np.genfromtxt(fname,delimiter=',')
    # Ensure data is a 2D array for consistent indexing
    if data.ndim == 1:
        data = data.reshape(1, -1)

    pos1_data = data[:, 0]
    power1_data = data[:, 1]
    pos2_data = data[:, 2]
    power2_data = data[:, 3]

    # --- Create Figure and Subplots ---
    fig, axs = plt.subplots(2, 3, figsize=(14, 10))
    fig.suptitle('Flicker Measurement Analysis', fontsize=16)

    # --- 1. Histogram for Position 1 ---
    ax1 = axs[0, 0]
    pos1_errors = pos1_data - zero1
    ax1.hist(pos1_errors, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title(f'Position 1 Error (Target: {zero1})')
    ax1.set_xlabel('Error from Target Position (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 2. Scatter Plot for Power 1 ---
    ax2 = axs[0, 1]
    x_axis = np.arange(len(power1_data))
    ax2.scatter(x_axis, power1_data, alpha=0.7, label='Measured Power')

    # Linear regression
    def linear_func(x, a, b):
        return a * x + b
    popt, _ = curve_fit(linear_func, x_axis, power1_data)
    a, b = popt
    ax2.plot(x_axis, linear_func(x_axis, a, b), 'r-', label=f'Fit: y={a:.4e}x + {b:.4f}')

    ax2.set_title('Power Readings at Position 1')
    ax2.set_xlabel('Measurement Index')
    ax2.set_ylabel('Power (W)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- 3. Histogram for Position 2 ---
    ax3 = axs[1, 0]
    pos2_errors = pos2_data - zero2
    ax3.hist(pos2_errors, bins=20, color='salmon', edgecolor='black')
    ax3.set_title(f'Position 2 Error (Target: {zero2})')
    ax3.set_xlabel('Error from Target Position (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # --- 4. Scatter Plot for Power 2 ---
    ax4 = axs[1, 1]
    x_axis_p2 = np.arange(len(power2_data))
    ax4.scatter(x_axis_p2, power2_data, alpha=0.7, color='green', label='Measured Power')

    # Linear regression
    popt2, _ = curve_fit(linear_func, x_axis_p2, power2_data)
    a2, b2 = popt2
    ax4.plot(x_axis_p2, linear_func(x_axis_p2, a2, b2), 'r-', label=f'Fit: y={a2:.4e}x + {b2:.4f}')

    ax4.set_title('Power Readings at Position 2')
    ax4.set_xlabel('Measurement Index')
    ax4.set_ylabel('Power (W)')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)

    # --- 5. Position Plot for Zero 1 ---
    ax5 = axs[0, 2]
    x_axis = np.arange(len(pos1_data))
    ax5.scatter(x_axis, pos1_data, alpha=0.7, label='Position Drift 1')

    # --- 5. Position Plot for Zero 2 ---
    ax6 = axs[1, 2]
    x_axis = np.arange(len(pos2_data))
    ax6.scatter(x_axis, pos2_data, alpha=0.7, label='Position Drift 2')

    # --- Display Plot ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() 
    
def makenewgraph(fname, zero1, zero2):
    """
    Generates four subplots to analyze the flicker measurement data.
    1. Histogram of position errors at zero1.
    2. Scatter plot and linear regression of power at zero1.
    3. Histogram of position errors at zero2.
    4. Scatter plot and linear regression of power at zero2.
    """
    # --- Data Extraction ---
    data = np.genfromtxt(fname,delimiter=',')
    # Ensure data is a 2D array for consistent indexing
    if data.ndim == 1:
        data = data.reshape(1, -1)

    pos1_data = data[:, 0]
    power1_data = data[:, 1]
    pos2_data = data[:, 2]
    power2_data = data[:, 3]

    # --- Create Figure and Subplots ---
    fig, axs = plt.subplots(2, 3, figsize=(14, 10))
    fig.suptitle('Flicker Measurement Analysis', fontsize=16)

    # --- 1. Histogram for Position 1 ---
    ax1 = axs[0, 0]
    lastpos1 = np.roll(pos1_data,1)
    lastpos1[0] = zero1
    pos1_errors = pos1_data - lastpos1
    ax1.hist(pos1_errors, bins=20, color='skyblue', edgecolor='black')
    ax1.set_title(f'Position 1 Error (Target: {zero1})')
    ax1.set_xlabel('Error from Target Position (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 2. Scatter Plot for Power 1 ---
    ax2 = axs[0, 1]
    x_axis = np.arange(len(power1_data))
    ax2.scatter(x_axis, power1_data, alpha=0.7, label='Measured Power')

    # Linear regression
    def linear_func(x, a, b):
        return a * x + b
    popt, _ = curve_fit(linear_func, x_axis, power1_data)
    a, b = popt
    ax2.plot(x_axis, linear_func(x_axis, a, b), 'r-', label=f'Fit: y={a:.4e}x + {b:.4f}')

    ax2.set_title('Power Readings at Position 1')
    ax2.set_xlabel('Measurement Index')
    ax2.set_ylabel('Power (W)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- 1. Histogram for Position 2 ---
    ax3 = axs[1, 0]
    lastpos2 = np.roll(pos1_data,1)
    lastpos2[0] = zero1
    pos1_errors = pos1_data - lastpos1
    ax3.hist(pos1_errors, bins=20, color='salmon', edgecolor='black')
    ax3.set_title(f'Position 2 Error (Target: {zero1})')
    ax3.set_xlabel('Error from Target Position (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, linestyle='--', alpha=0.6)

    # --- 4. Scatter Plot for Power 2 ---
    ax4 = axs[1, 1]
    x_axis_p2 = np.arange(len(power2_data))
    ax4.scatter(x_axis_p2, power2_data, alpha=0.7, color='green', label='Measured Power')

    # Linear regression
    popt2, _ = curve_fit(linear_func, x_axis_p2, power2_data)
    a2, b2 = popt2
    ax4.plot(x_axis_p2, linear_func(x_axis_p2, a2, b2), 'r-', label=f'Fit: y={a2:.4e}x + {b2:.4f}')

    ax4.set_title('Power Readings at Position 2')
    ax4.set_xlabel('Measurement Index')
    ax4.set_ylabel('Power (W)')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)

    # --- 5. Position Plot for Zero 1 ---
    ax5 = axs[0, 2]
    x_axis = np.arange(len(pos1_data))
    ax5.scatter(x_axis, pos1_data, alpha=0.7, label='Position Drift 1')

    # --- 5. Position Plot for Zero 2 ---
    ax6 = axs[1, 2]
    x_axis = np.arange(len(pos2_data))
    ax6.scatter(x_axis, pos2_data, alpha=0.7, label='Position Drift 2')

    # --- Display Plot ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()     

def parabola(x, a, x0, c):
    return a*(x-x0)**2 + c

def calcAllanDev(x):
    nPoints = np.size(x)

    allanDevs = np.zeros((int(nPoints/3),2))

    for intNum in np.arange(1,int(nPoints/3)+1):
        xTemp = np.mean(np.reshape(x[:int(nPoints/intNum)*intNum],[int(nPoints/intNum),intNum]),axis=1)
        allanDevs[intNum-1,1] = np.sqrt(np.mean(np.diff(xTemp)**2)/(2))
        allanDevs[intNum-1,0] = intNum

    return allanDevs

def analyseZerosLoop(fname, norm=False, analyseNorm=False):

    allData = np.genfromtxt(fname,delimiter=',')

    if norm == True:
        dataShape = np.shape(allData)
        nMeas = int((dataShape[1]-1)/7)

        reshapedData = np.zeros((dataShape[0],int(1+2*nMeas)))
        reshapedData[:,0] = allData[:,0]

        anglesColumnIdx = np.arange(1,int(1+7*nMeas),7)
        reshapedAnglesColumnIdx = np.arange(1,int(1+2*nMeas),2)

        reshapedData[:,reshapedAnglesColumnIdx] = allData[:,anglesColumnIdx]
        powersColumnIdx = np.arange(2,int(1+7*nMeas),7)
        powersNormColumnIdx = np.arange(5,int(1+7*nMeas),7)
        reshapedPowersColumnIdx = np.arange(2,int(1+2*nMeas),2)

        if analyseNorm == True:
            reshapedData[:,reshapedPowersColumnIdx] = allData[:,powersColumnIdx]/allData[:,powersNormColumnIdx]*np.mean(allData[:,powersNormColumnIdx])

        elif analyseNorm == False:
            reshapedData[:,reshapedPowersColumnIdx] = allData[:,powersColumnIdx]
        
        allData = reshapedData


    dataShape = np.shape(allData)
    lDataset = dataShape[0]
    print(lDataset)
    nDatasets = (dataShape[1]-1)/2
    print(dataShape)


    print(f'{nDatasets} total datasets')

    z1Guess = allData[int(lDataset/4),0]
    z2Guess = allData[int(3*lDataset/4),0]

    Cguess = np.min(allData[:,2])
    Aguess = (allData[0,2]-allData[int(lDataset/4),2])/((allData[0,0]-allData[int(lDataset/4),0])**2)

    print(z1Guess,z2Guess,Cguess,Aguess)

    results = np.zeros((int(nDatasets),6))

    for n in np.arange(nDatasets):
        data1 = allData[0:int(lDataset/2),(2*int(n)+1):(2*int(n)+3)]
        data2 = allData[int(lDataset/2):,(2*int(n)+1):(2*int(n)+3)]

        params1, cov1 = curve_fit(parabola,data1[:,0],data1[:,1],p0=[Aguess,z1Guess,Cguess],sigma=(0.05*data1[:,1]))
        params2, cov2 = curve_fit(parabola,data2[:,0],data2[:,1],p0=[Aguess,z2Guess,Cguess],sigma=(0.05*data2[:,1]))

        results[int(n),:] = np.array([params1[1],np.sqrt(cov1[1,1]),params2[1],np.sqrt(cov2[1,1]),np.abs(params2[1]-params1[1]),np.sqrt(cov1[1,1]+cov2[1,1])])

        if n==0:
            data1First = data1
            data2First = data2
            params1First = params1
            cov1First = cov1
            params2First = params2
            cov2First = cov2

            print(f'first fit: z1: {params1[1]} +/- {np.sqrt(cov1[1,1])}')
        
        if n==nDatasets-1:
            data1Last = data1
            data2Last = data2
            params1Last = params1
            cov1Last = cov1
            params2Last = params2
            cov2Last = cov2
            print(f'last fit: z1: {params1[1]} +/- {np.sqrt(cov1[1,1])}')


    print(f'zero1: {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.5}')
    print(f'zero2: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.5}')
    print(f'zero2-zero1: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.5}')

    print(f'zero2-zero1 co-std dev: {np.sqrt(np.abs(np.cov(np.transpose(results[:,[0,2]]))))}')
    print(f'zero2-zero1 correlation coeff: {np.corrcoef(np.transpose(results[:,[0,2]]))[0,1]:.5}')

    if nDatasets > 1:
        # fit zero positions to linear line
        linParams1, cov1 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,0])
        linParams2, cov2 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,2])
        linParams3, cov3 = curve_fit(lambda x,m,b: m*x+b, np.arange(nDatasets), results[:,4])

        print(f'zero1 slope: {linParams1[0]:.5}')
        print(f'zero2 slope: {linParams2[0]:.5}')
        print(f'mean slope: {(linParams1[0]+linParams2[0])/2:.5}')
        print(f'correction factor based on slope: {360/(360+(linParams1[0]+linParams2[0])/2)}')
        print(f'correction factor based on slope: {(360+(linParams1[0]+linParams2[0])/2)/360}')

        print(f'correction factor based on period: {180/(np.mean(results[:,4]))}')
        print(f'correction factor based on period: {(np.mean(results[:,4]))/180}')




    nbins=np.max([10,int(np.sqrt(nDatasets))])
    f,ax = plt.subplots(3,3,figsize =(16,10))
    ax[0,0].hist(results[:,0],bins=nbins,label=f'zero1: {np.mean(results[:,0]):.7} +/- {np.std(results[:,0]):.3}')
    ax[0,1].hist(results[:,2],bins=nbins,label=f'zero2: {np.mean(results[:,2]):.7} +/- {np.std(results[:,2]):.3}')
    ax[0,2].hist(results[:,4],bins=nbins,label=f'zero2-zero1: {np.mean(results[:,4]):.7} +/- {np.std(results[:,4]):.3}')
    ax[1,0].plot(results[:,0],results[:,2],'.',label=f'zero2-zero1 correlation coeff: {np.corrcoef(np.transpose(results[:,[0,2]]))[0,1]:.5}')
    ax[1,1].plot(results[:,0]-np.mean(results[:,0]),'.',label=f'zero1, mean fit error:{np.mean(results[:,1]):.3}')
    ax[1,1].plot(results[:,2]-np.mean(results[:,2]),'.',label=f'zero2, mean fit error:{np.mean(results[:,3]):.3}')
    if nDatasets > 1:
        ax[1,1].plot(linParams1[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 1: slope = {linParams1[0]:.3} +/- {np.sqrt(cov1[0,0]):.3}')
        ax[1,1].plot(linParams2[0]*(np.arange(nDatasets)-(nDatasets-1)/2),label=f'fit 2: slope = {linParams2[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')
    ax[1,2].plot(results[:,4],'.',label=f'data, mean fit error:{np.mean(results[:,5]):.3}')
    if nDatasets > 1:
        ax[1,2].plot(linParams3[0]*(np.arange(nDatasets))+linParams3[1],label=f'fit: slope = {linParams3[0]:.3} +/- {np.sqrt(cov2[0,0]):.3}')

    ax[0,0].legend()
    ax[0,1].legend()
    ax[0,2].legend()
    ax[1,0].legend()
    ax[1,1].legend()
    ax[1,2].legend()

    ax[0,0].set_title('zero 1 position distribution', fontweight='bold')
    ax[0,1].set_title('zero 2 position distribution', fontweight='bold')
    ax[0,2].set_title('zero1 - zero2 distance distribution', fontweight='bold')

    ax[1,0].set_title('zero 1 <--> zero 2 correlation', fontweight='bold')
    ax[1,1].set_title('zero positions \'time\' dependence', fontweight='bold')
    ax[1,2].set_title('zero1 - zero2 distance \'time\' dependence', fontweight='bold')

    ax[2,0].set_title('zero 1 fit performance', fontweight='bold')
    ax[2,1].set_title('zero 2 fit performance', fontweight='bold')

    ax[0,0].set_xlabel('zero 1 position (degrees)')
    ax[0,1].set_xlabel('zero 2 position (degrees)')
    ax[0,2].set_xlabel('zero 2 - zero 1 position (degrees)')
    ax[1,0].set_xlabel('zero 1 position (degrees)')
    ax[1,0].set_ylabel('zero 2 position (degrees)')
    ax[1,1].set_xlabel('dataset number')
    ax[1,1].set_ylabel('offset zero position (degrees)')
    ax[1,2].set_xlabel('dataset number')
    ax[1,2].set_ylabel('zero 2 - zero 1 position (degrees)')

    data1Dense = np.arange(data1First[0,0],data1First[-1,0],0.001)
    data2Dense = np.arange(data2First[0,0],data2First[-1,0],0.001)

    ax[2,0].plot(data1First[:,0],data1First[:,1],'.',label='first data set')
    ax[2,0].plot(data1Last[:,0],data1Last[:,1],'.',label='last data set')
    ax[2,0].plot(data1Dense,parabola(data1Dense,*params1First),label='first fit')
    ax[2,0].plot(data1Dense,parabola(data1Dense,*params1Last),label='last fit')
    ax[2,0].plot(data1First[:-1,0]+np.diff(data1First[:,0]),np.diff(data1First[:,1]),'.')
    ax[2,0].grid(which='major',linewidth='0.5',color='gray')

    ax[2,1].plot(data2First[:,0],data2First[:,1],'.',label='first data set')
    ax[2,1].plot(data2Last[:,0],data2Last[:,1],'.',label='last data set')
    ax[2,1].plot(data2Dense,parabola(data2Dense,*params2First),label='first fit')
    ax[2,1].plot(data2Dense,parabola(data2Dense,*params2Last),label='last fit')

    ax[2,0].legend()
    ax[2,1].legend()
    ax[2,0].set_xlabel('bellMotors\' stage position (degrees)')
    ax[2,1].set_xlabel('bellMotors\' stage position (degrees)')
    ax[2,0].set_ylabel('power (W)')
    ax[2,1].set_ylabel('power (W))')

    print(nDatasets)

    if nDatasets >= 6:
        # calculate and plot Allan deviations of zero positions and zero-position separation
        allanDevZ1 = calcAllanDev(results[:,0])
        allanDevZ2 = calcAllanDev(results[:,2])
        allanDevZsep = calcAllanDev(results[:,4])

        ax[2,2].plot(allanDevZ1[:,0],allanDevZ1[:,1],label='zero 1')
        ax[2,2].plot(allanDevZ2[:,0],allanDevZ2[:,1],label='zero 2')
        ax[2,2].plot(allanDevZsep[:,0],allanDevZsep[:,1],label='zeros\' separation')
        ax[2,2].set_xscale('log')
        ax[2,2].set_yscale('log')
        ax[2,2].grid(which='major',linewidth='0.5',color='gray')
        ax[2,2].grid(which='minor',linewidth='0.5',color='lightgray')
        ax[2,2].set_xlabel('number of measurements')
        ax[2,2].set_ylabel('position uncertainty (degrees)')
        ax[2,2].legend()
        ax[2,2].set_title('zero position Allan deviations', fontweight='bold')
    
    f.suptitle(fname+'\n',fontweight = 'bold',fontsize=14)

    if norm == True:
        if analyseNorm == False:
            f.suptitle(fname+'    raw data\n',fontweight = 'bold',fontsize=14)
        elif analyseNorm == True:
            f.suptitle(fname+'    normalized data\n',fontweight = 'bold',fontsize=14)


    plt.tight_layout()

    plt.show()

    return allData, results



def main():
    pmeter = setup_powermeter(pmAddr,1000)
    print(pmeter)
    stage.home('TestELL16')
    print("Homed Stage")
    time.sleep(1)
    filepath = './data/2025_06_30/powerCycles_'+str(int(time.time()))+'.csv'
    d=measFlickerLoop(filepath,stage,21.4051,201.4051,pmeter,1500,50) #fname,stage,zero1,zero2,pm,countNum,nLoops
    print(d)
    analyseFlickerLoop(filepath,21.4051,201.4051)
    stage.close()
    pmeter.close()
    print("Loop Completed Successfully!")
    
if __name__ == "__main__":
    main()
