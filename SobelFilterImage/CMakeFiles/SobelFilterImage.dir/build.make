# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\msys64\mingw64\bin\cmake.exe

# The command to remove a file.
RM = C:\msys64\mingw64\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage

# Include any dependencies generated for this target.
include CMakeFiles/SobelFilterImage.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SobelFilterImage.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SobelFilterImage.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SobelFilterImage.dir/flags.make

CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.obj: CMakeFiles/SobelFilterImage.dir/flags.make
CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.obj: CMakeFiles/SobelFilterImage.dir/includes_CXX.rsp
CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.obj: SobelFilterImage.cpp
CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.obj: CMakeFiles/SobelFilterImage.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.obj"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.obj -MF CMakeFiles\SobelFilterImage.dir\SobelFilterImage.cpp.obj.d -o CMakeFiles\SobelFilterImage.dir\SobelFilterImage.cpp.obj -c D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage\SobelFilterImage.cpp

CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.i"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage\SobelFilterImage.cpp > CMakeFiles\SobelFilterImage.dir\SobelFilterImage.cpp.i

CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.s"
	C:\msys64\mingw64\bin\c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage\SobelFilterImage.cpp -o CMakeFiles\SobelFilterImage.dir\SobelFilterImage.cpp.s

# Object files for target SobelFilterImage
SobelFilterImage_OBJECTS = \
"CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.obj"

# External object files for target SobelFilterImage
SobelFilterImage_EXTERNAL_OBJECTS =

bin/x86_64/Release/SobelFilterImage.exe: CMakeFiles/SobelFilterImage.dir/SobelFilterImage.cpp.obj
bin/x86_64/Release/SobelFilterImage.exe: CMakeFiles/SobelFilterImage.dir/build.make
bin/x86_64/Release/SobelFilterImage.exe: C:/Program\ Files\ (x86)/AMD\ APP\ SDK/3.0/lib/x86_64/libOpenCL.a
bin/x86_64/Release/SobelFilterImage.exe: CMakeFiles/SobelFilterImage.dir/linkLibs.rsp
bin/x86_64/Release/SobelFilterImage.exe: CMakeFiles/SobelFilterImage.dir/objects1.rsp
bin/x86_64/Release/SobelFilterImage.exe: CMakeFiles/SobelFilterImage.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin\x86_64\Release\SobelFilterImage.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\SobelFilterImage.dir\link.txt --verbose=$(VERBOSE)
	C:\msys64\mingw64\bin\cmake.exe -E copy_if_different D:/AGH/semestr_8/AAW/projekt/AAW_LINE_DETECTION/SobelFilterImage/SobelFilterImage_Kernels.cl D:/AGH/semestr_8/AAW/projekt/AAW_LINE_DETECTION/SobelFilterImage/bin/x86_64/Release/.
	C:\msys64\mingw64\bin\cmake.exe -E copy_if_different D:/AGH/semestr_8/AAW/projekt/AAW_LINE_DETECTION/SobelFilterImage/SobelFilterImage_Kernels.cl ./
	C:\msys64\mingw64\bin\cmake.exe -E copy_if_different D:/AGH/semestr_8/AAW/projekt/AAW_LINE_DETECTION/SobelFilterImage/SobelFilterImage_Input.bmp D:/AGH/semestr_8/AAW/projekt/AAW_LINE_DETECTION/SobelFilterImage/bin/x86_64/Release/.
	C:\msys64\mingw64\bin\cmake.exe -E copy_if_different D:/AGH/semestr_8/AAW/projekt/AAW_LINE_DETECTION/SobelFilterImage/SobelFilterImage_Input.bmp ./

# Rule to build all files generated by this target.
CMakeFiles/SobelFilterImage.dir/build: bin/x86_64/Release/SobelFilterImage.exe
.PHONY : CMakeFiles/SobelFilterImage.dir/build

CMakeFiles/SobelFilterImage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\SobelFilterImage.dir\cmake_clean.cmake
.PHONY : CMakeFiles/SobelFilterImage.dir/clean

CMakeFiles/SobelFilterImage.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage D:\AGH\semestr_8\AAW\projekt\AAW_LINE_DETECTION\SobelFilterImage\CMakeFiles\SobelFilterImage.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SobelFilterImage.dir/depend

