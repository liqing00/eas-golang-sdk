package easprediction

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"./tf_predict_protos"
	"./torch_predict_protos"
	"github.com/golang/protobuf/proto"
)

// PredictClient for accessing prediction service by creating a fixed size connection pool
// to perform the request through established persistent connections.
type PredictClient struct {
	retryCount         int
	maxConnectionCount int
	token              string
	// endpoint           interface{}
	vipSrvEndPoint   vipServerEndpoint
	gtwayEndPoint    gatewayEndpoint
	cacheSrvEndPoint cacheServerEndpoint
	timeout          time.Duration
	endpointType     string
	endpointName     string
	// modelName          string
	serviceName string
	stop        bool
	client      http.Client
	// transport          http.Transport
}

// NewPredictClient returns an instance of PredictClient
func NewPredictClient(endpointName string, serviceName string) *PredictClient {
	return &PredictClient{
		endpointName: endpointName,
		serviceName:  serviceName,
		// token:       token,
		retryCount: 5,
		timeout:    5000 * time.Millisecond,
		client: http.Client{
			Timeout: 5000 * time.Millisecond,
			Transport: &http.Transport{
				MaxIdleConns:        1000,
				MaxIdleConnsPerHost: 100,
				// ExpectContinueTimeout: 10 * time.Millisecond,
			},
		},
	}
}

// Init initialize client
func (p *PredictClient) Init() {
	if p.endpointType == "" || p.endpointType == "DEFAULT" {
		p.gtwayEndPoint = *newGatewayEndpoint(p.endpointName)
	} else if p.endpointType == "VIPSERVER" {
		p.vipSrvEndPoint = *newVipServerEndpoint(p.endpointName)
	} else if p.endpointType == "DIRECT" {
		p.cacheSrvEndPoint = *newCacheServerEndpoint(p.endpointName, p.serviceName)
	} else {
		defer fmt.Println("Code: 500, Message: Unsupported endpoint type: ", p.endpointType)
		panic(fmt.Errorf("Code: 500, Message: Unsupported endpoint type: %s", p.endpointType))
	}
	go p.syncHandler()
}

// syncHandler sync endpoint with server
func (p *PredictClient) syncHandler() {
	for true {
		if p.stop {
			break
		}
		if p.endpointType == "VIPSERVER" {
			p.vipSrvEndPoint.sync()
		} else if p.endpointType == "DIRECT" {
			p.cacheSrvEndPoint.sync()
		}
		time.Sleep(3 * time.Second)
	}
}

// SetEndpoint for client
func (p *PredictClient) SetEndpoint(endpointName string) {
	p.endpointName = endpointName
}

// SetEndpointType for client
func (p *PredictClient) SetEndpointType(endpointType string) {
	p.endpointType = endpointType
}

// SetToken function sets token for client
func (p *PredictClient) SetToken(token string) {
	p.token = token
}

// SetRetryCount for client
func (p *PredictClient) SetRetryCount(cnt int) {
	p.retryCount = cnt
}

// SetTimeout for client
func (p *PredictClient) SetTimeout(timeout int) {
	p.timeout = time.Duration(timeout) * time.Millisecond
	p.client.Timeout = p.timeout
}

// SetServiceName for client
func (p *PredictClient) SetServiceName(serviceName string) {
	p.serviceName = serviceName
}

// buildURI returns an url for request
func (p *PredictClient) buildURI() string {
	endName := p.endpointName
	if p.endpointType == "" || p.endpointType == "DEFAULT" {
		endName = p.gtwayEndPoint.Get()
	} else if p.endpointType == "VIPSERVER" {
		endName = p.vipSrvEndPoint.Get()
	} else if p.endpointType == "DIRECT" {
		endName = p.cacheSrvEndPoint.Get()
	}

	if p.serviceName[len(p.serviceName)-1] == '/' {
		p.serviceName = p.serviceName[:len(p.serviceName)-1]
	}
	return fmt.Sprintf("http://%s/api/predict/%s", endName, p.serviceName)
}

// predict function posts inputs rawData to server and get response as []byte{}
func (p *PredictClient) predict(rawData []byte) []byte {
	url := p.buildURI()
	req, _ := http.NewRequest("POST", url, bytes.NewReader(rawData))
	req.Header.Set("Content-Type", "application/octet-stream")
	if p.token != "" {
		req.Header.Set("Authorization", p.token)
	}
	for i := 0; i < p.retryCount; i++ {
		resp, err := p.client.Do(req)
		if err != nil {
			if i == p.retryCount-1 {
				panic(err)
			}
			continue
		}
		defer resp.Body.Close()
		if resp.StatusCode == 500 {
			if i != p.retryCount-1 {
				continue
			}
			panic(resp.Status)
		}
		// if resp.StatusCode != 200 {
		// 	fmt.Println(resp.Body)
		// 	panic(resp.Body)
		// }
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil || resp.StatusCode != 200 {
			fmt.Println(string(body))
			panic(err)
		}
		return body
	}
	return []byte{}
}

// TorchPredict function send input data and return PyTorch predicted result
func (p *PredictClient) TorchPredict(request TorchRequest) TorchResponse {
	reqdata, err := proto.Marshal(&request.RequestData)
	if err != nil {
		fmt.Println("Marshal error: ", err)
		panic(err)
	}

	body := p.predict(reqdata)
	bd := &torch_predict_protos.PredictResponse{}
	proto.Unmarshal(body, bd)
	rsp := &TorchResponse{*bd}

	return *rsp
}

// TfPredict function send input data and return TensorFlow predicted result
func (p *PredictClient) TfPredict(request TfRequest) TfResponse {
	reqdata, err := proto.Marshal(&request.RequestData)
	if err != nil {
		fmt.Println("Marshal error: ", err)
		panic(err)
	}

	body := p.predict(reqdata)
	bd := &tf_predict_protos.PredictResponse{}
	proto.Unmarshal(body, bd)
	rsp := &TfResponse{*bd}

	return *rsp
}
