/*
 *  json_decode.c
 *
 *  Created by Léa Strobino.
 *  Copyright 2018. All rights reserved.
 *
 */

#include <mex.h>
#include <string.h>
#include "jsmn.h"

enum error {
  ERROR_MINRHS,
  ERROR_MAXRHS,
  ERROR_INVALID_ARGUMENT,
  ERROR_MALLOC
};

char *json_str;

void error(enum error e) {
  switch (e) {
    case ERROR_MINRHS:
      mexErrMsgIdAndTxt("MATLAB:minrhs","Not enough input arguments.");
    case ERROR_MAXRHS:
      mexErrMsgIdAndTxt("MATLAB:maxrhs","Too many input arguments.");
    case ERROR_INVALID_ARGUMENT:
      mexErrMsgIdAndTxt("json_decode:InvalidArgument","Requires string input.");
    case ERROR_MALLOC:
      mexErrMsgIdAndTxt("json_decode:malloc","Insufficient free heap space.");
  }
}

void error_parse(const unsigned int character) {
  
  char str[256];
  unsigned int i;
  
  sprintf(str,"Parse error at character %u:\n",character+1);
  
  if (character > 40) {
    sprintf(str,"%s%.*s\n",str,80,json_str+character-40);
    sprintf(str,"%s                                        ",str);
  } else {
    sprintf(str,"%s%.*s\n",str,80,json_str);
    for (i=0; i<character; i++) sprintf(str,"%s ",str);
  }
  sprintf(str,"%s^",str);
  
  mexErrMsgIdAndTxt("json_decode:ParseError",str);
  
}

char *json_get_string(jsmntok_t *t) {
  
  char *str = mxMalloc(t->end-t->start+1);
  if (str == NULL) error(ERROR_MALLOC);
  
  memcpy(str,&json_str[t->start],t->end-t->start);
  str[t->end-t->start] = '\0';
  
  return str;
  
}

unsigned int json_parse_item(jsmntok_t *t, mxArray **obj) {
  
  mxArray **array, *item;
  mxChar *chars;
  mxClassID classID;
  mwSize n[2] = {1,1};
  char *str;
  int cat = 1;
  unsigned int i, j;
  
  switch (t->type) {
    
    case JSMN_OBJECT:
      *obj = mxCreateStructMatrix(1,1,0,NULL);
      for (i=j=0; i<t->size; i++) {
        str = json_get_string(t+1+j++);
        j += json_parse_item(t+1+j,&item);
        mxAddField(*obj,str);
        mxSetField(*obj,0,str,item);
        mxFree(str);
      }
      return j+1;
      
    case JSMN_ARRAY:
      array = mxMalloc(t->size*sizeof(mxArray*));
      for (i=j=0; i<t->size; i++) {
        j += json_parse_item(t+1+j,&array[i]);
        cat &= (mxGetNumberOfElements(array[i]) == 1);
        if (i == 0) {
          cat &= !mxIsChar(array[i]);
          classID = mxGetClassID(array[i]);
        } else {
          cat &= (classID == mxGetClassID(array[i]));
        }
      }
      if (cat && (mexCallMATLABWithTrap(1,obj,t->size,array,"vertcat") == NULL)) {
        for (i=0; i<t->size; i++) mxDestroyArray(array[i]);
      } else {
        *obj = mxCreateCellMatrix(t->size,1);
        for (i=0; i<t->size; i++) mxSetCell(*obj,i,array[i]);
      }
      mxFree(array);
      return j+1;
      
    case JSMN_PRIMITIVE:
      str = json_get_string(t);
      if ((str[0] != 't') && (str[0] != 'f') && (str[0] != 'n')) {
        *obj = mxCreateDoubleMatrix(1,1,mxREAL);
        double *value = mxGetData(*obj);
        sscanf(str,"%lg",value);
      } else if (strcmp(str,"true") == 0) {
        *obj = mxCreateLogicalScalar(1);
      } else if (strcmp(str,"false") == 0) {
        *obj = mxCreateLogicalScalar(0);
      } else if (strcmp(str,"null") == 0) {
        *obj = mxCreateDoubleScalar(mxGetNaN());
      } else {
        error_parse(t->start);
      }
      mxFree(str);
      return 1;
      
    case JSMN_STRING:
      n[1] = t->end-t->start;
      *obj = mxCreateCharArray(2,n);
      chars = mxGetChars(*obj);
      for (i=t->start,j=0; i<t->end; i++) {
        /* decode escaped characters */
        if (json_str[i] == '\\') {
          if (json_str[i+1] == 'u') {
            sscanf(&json_str[i+2],"%4hx",&chars[j++]);
            i += 5;
          } else {
            switch (json_str[i+1]) {
              case '"':
              case '\\':
              case '/':
                *(char*)&chars[j++] = json_str[i+1];
                break;
              case 'b':
                *(char*)&chars[j++] = '\b';
                break;
              case 'f':
                *(char*)&chars[j++] = '\f';
                break;
              case 'n':
                *(char*)&chars[j++] = '\n';
                break;
              case 'r':
                *(char*)&chars[j++] = '\r';
                break;
              case 't':
                *(char*)&chars[j++] = '\t';
                break;
            }
            i += 1;
          }
        } else {
          *(char*)&chars[j++] = json_str[i];
        }
      }
      mxSetN(*obj,j);
      return 1;
      
  }
  
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  if (nrhs < 1) error(ERROR_MINRHS);
  if (nrhs > 1) error(ERROR_MAXRHS);
  if (!mxIsChar(prhs[0])) error(ERROR_INVALID_ARGUMENT);
  
  /* get the JSON string */
  json_str = mxArrayToString(prhs[0]);
  size_t json_strlen = mxGetNumberOfElements(prhs[0]);
  
  /* prepare the parser */
  jsmn_parser p;
  jsmn_init(&p);
  
  /* allocate some tokens as a start */
  jsmntok_t *t;
  size_t n = 256;
  t = mxMalloc(n*sizeof(jsmntok_t));
  if (t == NULL) error(ERROR_MALLOC);
  
  int r;
  while (1) {
    r = jsmn_parse(&p,json_str,json_strlen,t,n);
    if (r != JSMN_ERROR_NOMEM) break;
    n *= 2;
    t = mxRealloc(t,n*sizeof(jsmntok_t));
    if (t == NULL) error(ERROR_MALLOC);
  }
  
  if (r < 0) error_parse(p.pos);
  
  /* create the MATLAB structure */
  if (r == 0) {
    plhs[0] = mxCreateStructMatrix(1,1,0,NULL);
  } else {
    r = json_parse_item(t,&plhs[0]);
  }
  
  /* free memory */
  mxFree(t);
  mxFree(json_str);
  
}
