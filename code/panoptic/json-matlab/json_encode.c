/*
 *  json_encode.c
 *
 *  Created by Léa Strobino.
 *  Copyright 2018. All rights reserved.
 *
 */

#include <mex.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>

#define BUFFER_SIZE 1048576

enum error {
  ERROR_MINRHS,
  ERROR_MAXRHS,
  ERROR_MALLOC,
  ERROR_ENCODING
};

char *json_str;
size_t json_strlen, json_strpos;

void error(enum error e) {
  switch (e) {
    case ERROR_MINRHS:
      mexErrMsgIdAndTxt("MATLAB:minrhs","Not enough input arguments.");
    case ERROR_MAXRHS:
      mexErrMsgIdAndTxt("MATLAB:maxrhs","Too many input arguments.");
    case ERROR_MALLOC:
      mexErrMsgIdAndTxt("json_encode:malloc","Insufficient free heap space.");
    case ERROR_ENCODING:
      mexErrMsgIdAndTxt("json_encode:sprintf:EncodingError","An encoding error occurred.");
  }
}

void error_unsupported_class(const mxArray *obj) {
  char str[256];
  sprintf(str,"Unsupported class: %s.",mxGetClassName(obj));
  mexErrMsgIdAndTxt("json_encode:UnsupportedClass",str);
}

void json_str_realloc() {
  json_strlen += BUFFER_SIZE;
  json_str = mxRealloc(json_str,json_strlen);
  if (json_str == NULL) error(ERROR_MALLOC);
}

void json_append_char(char c) {
  if (json_strlen-json_strpos < 2) json_str_realloc();
  json_str[json_strpos++] = c;
}

#define json_append_float(value) { \
  if (mxIsNaN(value)) json_append_string("null,"); \
  else if (mxIsInf(value)) { \
    if (value < 0) json_append_string("\"-Inf\","); \
    else json_append_string("\"Inf\","); \
  } else json_append_string("%.16g,",value); \
}

void json_append_string(char *format, ...) {
  int n;
  va_list args;
  va_start(args,format);
  while (1) {
    n = vsnprintf(&json_str[json_strpos],json_strlen-json_strpos,format,args);
    if (n < 0) error(ERROR_ENCODING);
    if (n < json_strlen-json_strpos) break;
    json_str_realloc();
  }
  va_end(args);
  json_strpos += n;
}

void json_encode_item(const mxArray *obj) {
  
  mxArray *item;
  mxChar *chars;
  unsigned int field, nfields;
  const char *fieldname;
  mxLogical *logical_ptr;
  double *double_ptr;
  float *single_ptr;
  uint8_t *uint8_ptr;
  int8_t *int8_ptr;
  uint16_t *uint16_ptr;
  int16_t *int16_ptr;
  uint32_t *uint32_ptr;
  int32_t *int32_ptr;
  uint64_t *uint64_ptr;
  int64_t *int64_ptr;
  unsigned int i, n;
  
  n = mxGetNumberOfElements(obj);
  
  if (mxIsChar(obj)) {
    
    json_append_char('"');
    chars = mxGetChars(obj);
    for (i=0; i<n; i++) {
      switch (chars[i]) {
        case '"':
        case '\\':
        case '/':
          json_append_char('\\');
          json_append_char(*(char*)&chars[i]);
          break;
        case '\b':
          json_append_string("\\b");
          break;
        case '\f':
          json_append_string("\\f");
          break;
        case '\n':
          json_append_string("\\n");
          break;
        case '\r':
          json_append_string("\\r");
          break;
        case '\t':
          json_append_string("\\t");
          break;
        default:
          if ((chars[i] < 32) || (chars[i] > 126)) json_append_string("\\u%04hx",chars[i]);
          else json_append_char(*(char*)&chars[i]);
      }
    }
    json_append_char('"');
    
  } else if (n == 0) {
    
    json_append_string("[]");
    
  } else {
    
    if (n > 1) json_append_char('[');
    
    switch (mxGetClassID(obj)) {
      
      case mxSTRUCT_CLASS:
        nfields = mxGetNumberOfFields(obj);
        for (i=0; i<n; i++) {
          json_append_char('{');
          for (field=0; field<nfields; field++) {
            fieldname = mxGetFieldNameByNumber(obj,field);
            item = mxGetFieldByNumber(obj,i,field);
            if (item != NULL) {
              json_append_string("\"%s\":",fieldname);
              json_encode_item(item);
              json_append_char(',');
            }
          }
          if (nfields > 0) json_strpos--;
          json_append_string("},");
        }
        break;
        
      case mxCELL_CLASS:
        for (i=0; i<n; i++) {
          json_encode_item(mxGetCell(obj,i));
          json_append_char(',');
        }
        break;
        
      case mxLOGICAL_CLASS:
        logical_ptr = mxGetData(obj);
        for (i=0; i<n; i++) {
          if (logical_ptr[i]) {
            json_append_string("true,");
          } else {
            json_append_string("false,");
          }
        }
        break;
        
      case mxDOUBLE_CLASS:
        double_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_float(double_ptr[i]);
        break;
        
      case mxSINGLE_CLASS:
        single_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_float(single_ptr[i]);
        break;
        
      case mxINT8_CLASS:
        int8_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_string("%i,",int8_ptr[i]);
        break;
        
      case mxUINT8_CLASS:
        uint8_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_string("%u,",uint8_ptr[i]);
        break;
        
      case mxINT16_CLASS:
        int16_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_string("%i,",int16_ptr[i]);
        break;
        
      case mxUINT16_CLASS:
        uint16_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_string("%u,",uint16_ptr[i]);
        break;
        
      case mxINT32_CLASS:
        int32_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_string("%i,",int32_ptr[i]);
        break;
        
      case mxUINT32_CLASS:
        uint32_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_string("%u,",uint32_ptr[i]);
        break;
        
      case mxINT64_CLASS:
        int64_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_string("%lli,",int64_ptr[i]);
        break;
        
      case mxUINT64_CLASS:
        uint64_ptr = mxGetData(obj);
        for (i=0; i<n; i++) json_append_string("%llu,",uint64_ptr[i]);
        break;
        
      default:
        error_unsupported_class(obj);
        
    }
    
    if (json_strpos) json_strpos--;
    if (n > 1) json_append_char(']');
    
  }
  
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  if (nrhs < 1) error(ERROR_MINRHS);
  if (nrhs > 1) error(ERROR_MAXRHS);
  
  json_str = mxMalloc(BUFFER_SIZE);
  if (json_str == NULL) error(ERROR_MALLOC);
  json_strlen = BUFFER_SIZE;
  json_strpos = 0;
  
  json_encode_item(prhs[0]);
  
  json_str[json_strpos] = '\0';
  plhs[0] = mxCreateString(json_str);
  mxFree(json_str);
  
}
