
// Store the template [from ascii file]	
   templ.pushfile(ID,thePixelTemp_);
.
.
.
.

//Inside cluster loop

// Interpolate from the track direction, apply pixel truncation
       templ.interpolate(ID, cotalpha, cotbeta, -1.f);
       maxpix = templ.pixmax();
// Truncate pixels and sum columns and rows

		for(i=0; i<TYSIZE; ++i) {
		   clustpy[i] = 0.;
		   for(j=0; j<TXSIZE; ++j) {
		      if(cluster[j][i] > maxpix) cluster[j][i] = maxpix;
		      clustpy[i] += cluster[j][i];
		   }
		   if(clustpy[i]>0.){hp[4]->Fill((double)clustpy[i], 1.);}
		}
   // next, identify the y-cluster ends, count total pixels, nypix, and logical pixels, logypx
   
       fypix=-1;
       nypix=0;
       lypix=0;
       for(i=0; i<TYSIZE; ++i) {
          if(clustpy[i] > 0.f) {
             if(fypix == -1) {fypix = i;}
             ++nypix;
             lypix = i;
          }
        }

   // next, center the cluster on the input array
   
        midpix = (fypix+lypix)/2;
        shifty = 10 - midpix;
   
//  If OK, shift everything   
   
        if(shifty > 0) {
           for(i=lypix; i>=fypix; --i) {
              clustpy[i+shifty] = clustpy[i];
              clustpy[i] = 0.;
           }
        } else if (shifty < 0) {
          for(i=fypix; i<=lypix; ++i) {
              clustpy[i+shifty] = clustpy[i];
              clustpy[i] = 0.;
            }
        }

        yhit+=shifty*ysize;
		for(i=0; i<TYSIZE; ++i) {
		   *fTrainingy << clustpy[i]/200000.f << ", ";
		}
		*fTrainingy << cotalpha/0.3f << ", " << cotbeta/9.47f << ", " << qclust/100000.f << ", " << yhit/(1.5f*ysize) << endl;
		pp[1]->Fill((double)cotbeta, (double)nypix);
// Do the template analysis on the cluster 
		for(j=0; j<TXSIZE; ++j) {
		   clustpx[j] = 0.;
		   for(i=0; i<TYSIZE; ++i) {
		      clustpx[j] += cluster[j][i];
		   }
		   if(clustpx[j]>0.){hp[6]->Fill((double)clustpx[j], 1.);}
		}

   // next, identify the x-cluster ends, count total pixels, nxpix, and logical pixels, logypx
   
       fxpix=-1;
       nxpix=0;
       lxpix=0;
       for(j=0; j<TXSIZE; ++j) {
          if(clustpx[j] > 0.f) {
             if(fxpix == -1) {fxpix = j;}
             ++nxpix;
             lxpix = j;
          }
        }
   // next, center the cluster on the input array
   
        midpix = (fxpix+lxpix)/2;
        shiftx = 6 - midpix;
   
//  If OK, shift everything   
   
        if(shiftx > 0) {
           for(j=lxpix; j>=fxpix; --j) {
              clustpx[j+shiftx] = clustpx[j];
              clustpx[j] = 0.;
           }
        } else if (shiftx < 0) {
          for(j=fxpix; j<=lxpix; ++j) {
              clustpx[j+shiftx] = clustpx[j];
              clustpx[j] = 0.;
            }
        }
        xhit+=shiftx*xsize;
		for(j=0; j<TXSIZE; ++j) {
		   *fTrainingx << clustpx[j]/200000.f << ", ";
		}
		*fTrainingx << cotalpha/0.3f << ", " << cotbeta/9.47f << ", " << qclust/200000.f << ", " << xhit/(1.5f*xsize) << endl;
		pp[2]->Fill((double)cotalpha, (double)nxpix);

	   }
